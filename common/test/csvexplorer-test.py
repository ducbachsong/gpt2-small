"""test — run csvexplorer over CSVs written here, and print what happened.

    python test/csvexplorer-test.py

Everything is local: the CSVs are written to a temporary folder, read back
through DuckDB, and the page is checked twice — once drawn headlessly with
Streamlit's own AppTest, which starts no server and uses no port, and once
started for real on a free port this test picks and prints. No network,
nothing left behind, and your own dashboard on 8501 is left alone.

The CSVs are two sensor logs, not training logs: the module must not care
what the numbers mean.

Each case is one function, run in order by main(); the first one opens the
CSVs the middle ones read. Every case prints what it found, so a failure
shows you where it stopped instead of just saying False.
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

# The module under test lives one folder up, next to this test folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csvexplorer
from csvexplorer import (aggregate_data, plot_data, chart_image,
                        compare_two_sources, default_x_column, find_csv_files,
                        group_names, line_chart, list_columns,
                        list_sources, numeric_columns, wide_by_source,
                        load_csvs, run_sql, summary_table)

MODULE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "csvexplorer.py")
BEFORE_ROWS = 200       # times 1..200
AFTER_ROWS = 100        # times 1, 3, 5 ... 199: half as many, so the two
                        # files do not share every reading
COOLER_BY = 0.1         # how much colder every "after" reading is
START_SECONDS = 90      # how long the real Streamlit server may take to answer


# ── starting and stopping a real server ─────────────────────────────────────
def free_port():
    """A port nothing is listening on right now.

    In : nothing.
    Out: int   e.g. 51234. Asked of the operating system, so this test never
               lands on the 8501 a dashboard of your own may be using.
    """
    with socket.socket() as looking:
        looking.bind(("127.0.0.1", 0))
        return looking.getsockname()[1]


def read_url(url, timeout=5):
    """Ask a URL for its page.

    In : url      str   e.g. "http://127.0.0.1:51234/_stcore/health"
         timeout  float seconds to wait for an answer
    Out: (status, text)  e.g. (200, "ok")
         None            nothing answered: no server there
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as answer:
            return answer.status, answer.read().decode(errors="replace")
    except Exception:
        return None


def stop_process_tree(process):
    """Stop the launcher and the Streamlit server it started.

    In : process  the launcher from subprocess.Popen
    Out: None
    """
    if os.name == "nt":
        # The server is a child of the launcher, so the whole tree has to go.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True)
    else:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()


# ── the CSVs this test reads ────────────────────────────────────────────────
def temperature_at(time):
    """The temperature the "before" log holds at `time`: 20.02 up to 24.00."""
    return 20.0 + 0.02 * time


def write_test_csvs(folder):
    """Write before.csv and after.csv, two logs of a made-up sensor.

    after.csv holds only the odd times, and one column before.csv never had,
    so lining the two up is not a matter of matching rows.

    In : folder  str   an empty folder to write into
    Out: None
    """
    with open(os.path.join(folder, "before.csv"), "w", encoding="utf-8",
              newline="") as csv_file:
        csv_file.write("time,temperature,humidity\n")
        for time in range(1, BEFORE_ROWS + 1):
            csv_file.write(f"{time},{temperature_at(time)},{50 - 0.05 * time}\n")

    with open(os.path.join(folder, "after.csv"), "w", encoding="utf-8",
              newline="") as csv_file:
        csv_file.write("time,temperature,humidity,battery\n")
        for point in range(AFTER_ROWS):
            time = 2 * point + 1                    # 1, 3, 5, ...
            csv_file.write(f"{time},{temperature_at(time) - COOLER_BY},"
                           f"{50 - 0.05 * time + 1},0.98\n")


# ── the cases ───────────────────────────────────────────────────────────────
def test_finds_the_files(folder):
    """A folder, a glob, a list and a ";" string all find the same two CSVs."""
    print("\n=== 1. finding the CSV files ===")
    by_folder = find_csv_files(folder)
    by_glob = find_csv_files(os.path.join(folder, "*.csv"))
    by_list = find_csv_files([os.path.join(folder, "before.csv"),
                              os.path.join(folder, "after.csv")])
    by_text = find_csv_files(f"{os.path.join(folder, 'before.csv')};"
                             f"{os.path.join(folder, 'after.csv')}")

    print(f"    {[os.path.basename(path) for path in by_folder]}")
    assert len(by_folder) == 2, f"a folder found {by_folder}"
    assert by_glob == by_folder == by_list == by_text, \
        "the four ways of naming the same files disagree"
    assert find_csv_files(os.path.join(folder, "nothing-*.csv")) == [], \
        "a pattern matching nothing found something"


def test_one_table_for_every_file(folder):
    """load_csvs() reads both CSVs into one table, columns lined up by name."""
    print("\n=== 2. the table ===")
    load_csvs(folder)

    columns = list_columns()
    print(f"    sources: {list_sources()}")
    print(f"    columns: {list(columns)}")
    assert list_sources() == ["after", "before"], "the sources are named wrong"
    for wanted in (csvexplorer.SOURCE_COLUMN, csvexplorer.ROW_COLUMN, "time",
                   "temperature", "humidity", "battery"):
        assert wanted in columns, f"{wanted} is missing from the table"

    # The table name is plain enough to write SQL against without quoting.
    counted = run_sql(f"SELECT source, count(*) AS rows FROM "
                      f"{csvexplorer.TABLE_NAME} GROUP BY source ORDER BY source")
    assert list(counted["rows"]) == [AFTER_ROWS, BEFORE_ROWS], \
        f"wrong row counts: {list(counted['rows'])}"

    # Only after.csv has a battery column, so before is NULL there.
    missing = run_sql("SELECT count(*) AS n FROM data "
                      "WHERE source = 'before' AND battery IS NULL")
    assert int(missing["n"][0]) == BEFORE_ROWS, \
        "the extra column of one file did not line up as NULL for the other"

    assert default_x_column() == "time", \
        f"the x axis picked itself wrong: {default_x_column()}"
    assert set(numeric_columns()) >= {"time", "temperature", "humidity"}, \
        f"number columns missing: {numeric_columns()}"
    print(f"    x axis: {default_x_column()}, numbers: {numeric_columns()}")


def test_chart_data_smoothing_and_thinning():
    """plot_data() gives the logged values, smoothed and thinned on request."""
    print("\n=== 3. the points to draw ===")
    frame = plot_data("time", ["temperature"], ["before"])
    assert list(frame.columns) == ["source", "value_column", "x", "value"], \
        f"wrong columns {list(frame.columns)}"
    assert len(frame) == BEFORE_ROWS, f"{len(frame)} points, not {BEFORE_ROWS}"
    assert abs(frame["value"][0] - temperature_at(1)) < 1e-9, \
        "the first point is not the value in the file"
    print(f"    {len(frame)} points, first {frame['value'][0]:.3f}, "
          f"last {frame['value'].iloc[-1]:.3f}")

    smoothed = plot_data("time", ["temperature"], ["before"], smoothing=2)
    expected = (temperature_at(4) + temperature_at(5)) / 2
    assert abs(smoothed["value"][4] - expected) < 1e-9, \
        "smoothing is not a rolling mean of the last two points"
    print(f"    smoothed over 2 points: {smoothed['value'][4]:.4f} "
          f"(mean of {temperature_at(4):.3f} and {temperature_at(5):.3f})")

    thinned = plot_data("time", ["temperature"], ["before"], max_points=50)
    print(f"    thinned to {len(thinned)} points from {BEFORE_ROWS}")
    assert len(thinned) <= 51, f"{len(thinned)} points, more than asked for"
    assert thinned["x"].iloc[-1] == BEFORE_ROWS, \
        "thinning dropped the end of the data"

    both = plot_data("time", ["temperature", "humidity"])
    assert set(both["value_column"]) == {"temperature", "humidity"}, \
        "a value column is missing"
    assert set(both["source"]) == {"before", "after"}, "a source is missing"


def test_summary_of_each_source():
    """summary_table() says where each source started, ended and went."""
    print("\n=== 4. the summary ===")
    summary = summary_table("time", ["temperature"])
    print(summary.to_string(index=False))

    before_row = summary[summary["source"] == "before"].iloc[0]
    assert int(before_row["points"]) == BEFORE_ROWS, "wrong point count"
    assert abs(before_row["first_value"] - temperature_at(1)) < 1e-9, \
        "first_value is not the value at the first x"
    assert abs(before_row["last_value"] - temperature_at(BEFORE_ROWS)) < 1e-9, \
        "last_value is not the value at the last x"
    assert abs(before_row["smallest"] - temperature_at(1)) < 1e-9
    assert abs(before_row["largest"] - temperature_at(BEFORE_ROWS)) < 1e-9

    after_row = summary[summary["source"] == "after"].iloc[0]
    assert after_row["last_value"] < before_row["last_value"], \
        "the two sources look the same, so the summary compares nothing"


def test_two_sources_side_by_side():
    """compare_two_sources() lines up files with different x values."""
    print("\n=== 5. after against before ===")
    comparison = compare_two_sources("after", "before", "temperature", "time")
    print(comparison.head(3).to_string(index=False))

    assert list(comparison.columns) == ["x", "after", "before", "difference"], \
        f"wrong columns {list(comparison.columns)}"
    assert len(comparison) == AFTER_ROWS, \
        f"{len(comparison)} rows, not one per point of after"
    assert comparison["difference"].notna().all(), "a point matched nothing"
    assert (comparison["difference"] + COOLER_BY).abs().max() < 1e-9, \
        f"the difference is not the {COOLER_BY} the files were written with"

    # The other way round, before has times after never logged, so those match
    # the reading before: that is what ASOF is for.
    other_way = compare_two_sources("before", "after", "temperature", "time")
    assert len(other_way) == BEFORE_ROWS, "ASOF dropped points of before"
    assert other_way["difference"].notna().all(), "a before point matched nothing"
    print(f"    the other way round: {len(other_way)} rows, "
          f"difference {other_way['difference'].min():.3f} to "
          f"{other_way['difference'].max():.3f}")


def test_one_column_per_source_is_chart_shaped():
    """wide_by_source() gives a frame a chart can draw as it is."""
    print("\n=== 6. the shape charts want ===")
    frame = plot_data("time", ["temperature"])
    wide = wide_by_source(frame, "temperature")
    print(wide.head(3).to_string())
    assert list(wide.columns) == ["after", "before"], \
        f"wrong columns {list(wide.columns)}"
    assert wide.index.name == "x", "the index is not the x axis"
    assert abs(wide["before"].iloc[0] - temperature_at(1)) < 1e-9


def test_csv_that_names_its_own_sources():
    """A CSV with a source column of its own holds several sources in one file.

    Written to a folder of its own, so the cases before it keep their two
    CSVs and nothing else.
    """
    print("\n=== 7. one file, two sources ===")
    one_file_folder = tempfile.mkdtemp(prefix="csvexplorer-one-file-")
    try:
        with open(os.path.join(one_file_folder, "everything.csv"), "w",
                  encoding="utf-8", newline="") as csv_file:
            csv_file.write("source,time,temperature\n")
            for source_name, offset in (("indoors", 0.0), ("outdoors", -5.0)):
                for time in range(1, 11):
                    csv_file.write(f"{source_name},{time},"
                                   f"{temperature_at(time) + offset}\n")

        load_csvs(one_file_folder)
        print(f"    sources from one file: {list_sources()}")
        assert list_sources() == ["indoors", "outdoors"], \
            "the source column in the CSV was not used"
        assert len(plot_data("time", ["temperature"], ["outdoors"])) == 10
    finally:
        shutil.rmtree(one_file_folder, ignore_errors=True)


def test_same_file_name_in_two_folders():
    """Two files called the same thing keep their folder in the source name."""
    print("\n=== 8. two files with one name ===")
    twice_folder = tempfile.mkdtemp(prefix="csvexplorer-twice-")
    try:
        for folder_name in ("kitchen", "garden"):
            sensor_folder = os.path.join(twice_folder, folder_name)
            os.makedirs(sensor_folder, exist_ok=True)
            with open(os.path.join(sensor_folder, "log.csv"), "w",
                      encoding="utf-8", newline="") as csv_file:
                csv_file.write("time,temperature\n1,20.0\n2,20.1\n")

        load_csvs(twice_folder)
        print(f"    sources: {list_sources()}")
        assert list_sources() == ["garden/log", "kitchen/log"], \
            "same-named files did not keep their folders apart"
    finally:
        shutil.rmtree(twice_folder, ignore_errors=True)


def test_files_of_one_setting_become_one_line():
    """Repeats of one setting are combined into a mean, with a band.

    Three files log 1.0, 2.0 and 3.0 at the same times, so the mean is 2.0,
    the standard deviation is exactly 1.0, and every band is known.
    """
    print("\n=== 9. three repeats, one line ===")
    seeds_folder = tempfile.mkdtemp(prefix="csvexplorer-seeds-")
    try:
        for number, value in enumerate((1.0, 2.0, 3.0), start=1):
            with open(os.path.join(seeds_folder, f"run-{number}.csv"), "w",
                      encoding="utf-8", newline="") as csv_file:
                csv_file.write("time,loss\n")
                for time in range(1, 6):
                    csv_file.write(f"{time},{value}\n")
        # One file of another setting, which must stay on its own.
        with open(os.path.join(seeds_folder, "tuned-1.csv"), "w",
                  encoding="utf-8", newline="") as csv_file:
            csv_file.write("time,loss\n")
            for time in range(1, 6):
                csv_file.write(f"{time},0.5\n")

        load_csvs(seeds_folder)
        groups = group_names(list_sources())
        print(f"    grouped: {groups}")
        assert groups == {"run-1": "run", "run-2": "run", "run-3": "run",
                          "tuned-1": "tuned"}, "the names were grouped wrong"

        combined = aggregate_data("time", ["loss"])
        print(combined.head(3).to_string(index=False))
        assert set(combined["group_name"]) == {"run", "tuned"}, \
            "the repeats did not become one line each"

        run_row = combined[combined["group_name"] == "run"].iloc[0]
        assert int(run_row["files"]) == 3, "the mean is not over three files"
        assert abs(run_row["value"] - 2.0) < 1e-9, "the mean is wrong"
        assert abs(run_row["low"] - 1.0) < 1e-9 and \
               abs(run_row["high"] - 3.0) < 1e-9, \
               "the standard deviation band is wrong"

        # A file on its own spreads by nothing, not by NULL.
        tuned_row = combined[combined["group_name"] == "tuned"].iloc[0]
        assert abs(tuned_row["low"] - tuned_row["high"]) < 1e-9, \
            "one file on its own came back with a band"

        for band, low, high in (("min to max", 1.0, 3.0),
                                ("quartiles", 1.5, 2.5),
                                ("standard error", 2.0 - 1 / 3 ** 0.5,
                                 2.0 + 1 / 3 ** 0.5)):
            row = aggregate_data("time", ["loss"], band=band).iloc[0]
            print(f"    {band}: {row['low']:.3f} to {row['high']:.3f}")
            assert abs(row["low"] - low) < 1e-9 and abs(row["high"] - high) < 1e-9, \
                f"the {band} band is wrong"

        without = aggregate_data("time", ["loss"], band="none")
        assert "low" not in without.columns, "band 'none' still drew a band"
    finally:
        shutil.rmtree(seeds_folder, ignore_errors=True)


def test_chart_and_the_files_it_saves():
    """line_chart() builds the figure, chart_image() writes png, svg and pdf."""
    print("\n=== 10. the figure ===")
    frame = plot_data("time", ["temperature"], ["before", "after"])
    chart = line_chart(frame, title="a temperature", x_title="time",
                       y_title="°C", markers=True, log_y=False)
    drawn = chart.to_dict()
    assert drawn["layer"], "the chart has no layers"
    assert drawn["title"] == "a temperature", "the title did not stick"

    for image_format, first_bytes in (("png", b"\x89PNG"), ("svg", b"<svg"),
                                      ("pdf", b"%PDF")):
        written = chart_image(chart, image_format, dpi=150)
        print(f"    {image_format}: {len(written):,} bytes, "
              f"starts {written[:4]!r}")
        assert written.startswith(first_bytes), \
            f"the {image_format} file does not look like one"

    # A combined frame carries low and high, which become a shaded band.
    banded = line_chart(plot_data("time", ["temperature"]).assign(
        low=lambda rows: rows["value"] - 1, high=lambda rows: rows["value"] + 1))
    assert len(banded.to_dict()["layer"]) == 2, "the band layer is missing"
    print("    a band adds a second layer")


def test_dashboard_draws(folder):
    """The Streamlit page runs end to end, headlessly, over the same CSVs."""
    print("\n=== 11. the Streamlit page ===")
    from streamlit.testing.v1 import AppTest

    os.environ[csvexplorer.PATHS_ENV] = folder     # which CSVs the page reads
    page = AppTest.from_file(MODULE_PATH, default_timeout=120)
    page.run()

    if page.exception:
        raise AssertionError(f"the page raised: {page.exception[0].value}")
    headings = [heading.value for heading in page.subheader]
    print(f"    title {page.title[0].value!r}, sections {headings}")
    assert page.title[0].value == csvexplorer.PAGE_TITLE, "wrong page title"
    assert "temperature" in headings and "Summary" in headings, \
        "the page drew no chart or no summary"

    # The Files picker: both CSVs found and loaded, and one can be dropped.
    files_picked = page.get_by_key("files")
    print(f"    files offered: {list(files_picked.options)}")
    assert sorted(files_picked.options) == ["after", "before"], \
        "the Files picker does not offer both CSVs"
    page.get_by_key("files").set_value(["before"]).run()
    if page.exception:
        raise AssertionError(f"dropping a file raised: {page.exception[0].value}")
    assert page.get_by_key("sources").options == ["before"], \
        "dropping a file left its source behind"
    page.get_by_key("files").set_value(["after", "before"]).run()

    # The filter and its buttons, for when there are many to pick from.
    page.get_by_key("values_filter").set_value("hum").run()
    page.get_by_key("values_only").click().run()
    if page.exception:
        raise AssertionError(f"filtering raised: {page.exception[0].value}")
    print(f"    after filtering values by 'hum': {page.get_by_key('values').value}")
    assert page.get_by_key("values").value == ["humidity"], \
        "'Only these' did not narrow the values to the filter"
    page.get_by_key("values_filter").set_value("").run()
    page.get_by_key("values").set_value(["temperature", "humidity", "battery"]).run()
    assert "Difference: after minus before" in headings, \
        "two sources were chosen but the difference section is missing"
    assert len(page.dataframe) >= 2, "the summary and difference tables are missing"

    # Move a control, as a person would, and draw again.
    page.get_by_key("smoothing").set_value(10).run()
    page.get_by_key("sources").set_value(["before"]).run()      # one source
    if page.exception:
        raise AssertionError(f"the page raised after a change: "
                             f"{page.exception[0].value}")
    headings = [heading.value for heading in page.subheader]
    print(f"    after choosing one source: {headings}")
    assert "Difference: after minus before" not in headings, \
        "the difference section stayed with only one source chosen"
    os.environ.pop(csvexplorer.PATHS_ENV, None)


def test_two_files_in_one_diagram(folder):
    """Both files, and any mix of their columns, drawn as lines in one chart."""
    print("\n=== 12. everything in one chart ===")
    from streamlit.testing.v1 import AppTest

    os.environ[csvexplorer.PATHS_ENV] = folder
    page = AppTest.from_file(MODULE_PATH, default_timeout=120)
    page.run()
    page.get_by_key("layout").set_value("everything in one").run()
    if page.exception:
        raise AssertionError(f"the page raised: {page.exception[0].value}")

    headings = [heading.value for heading in page.subheader]
    offered_lines = list(page.get_by_key("lines").options)
    print(f"    sections {headings}")
    print(f"    lines offered: {offered_lines}")
    assert "All series" in headings, "the one-chart section is missing"
    for wanted in (f"before{csvexplorer.LINE_SEPARATOR}temperature",
                   f"after{csvexplorer.LINE_SEPARATOR}humidity"):
        assert wanted in offered_lines, f"{wanted} is not on offer"

    # Two lines from two different files, and two different columns at that.
    two_lines = [f"before{csvexplorer.LINE_SEPARATOR}temperature",
                 f"after{csvexplorer.LINE_SEPARATOR}humidity"]
    page.get_by_key("lines").set_value(two_lines).run()
    if page.exception:
        raise AssertionError(f"the page raised on two lines: "
                             f"{page.exception[0].value}")
    print(f"    drew {two_lines}")

    # The other chart kinds and the two ways of putting values on one scale.
    for chart_kind in ("area", "scatter"):
        page.get_by_key("chart_kind").set_value(chart_kind).run()
        if page.exception:
            raise AssertionError(f"the {chart_kind} chart raised: "
                                 f"{page.exception[0].value}")
    for shown_as in ("minus first value", "percent from first"):
        page.get_by_key("shown_as").set_value(shown_as).run()
        if page.exception:
            raise AssertionError(f"{shown_as!r} raised: "
                                 f"{page.exception[0].value}")
    print("    line, area and scatter drawn; raw, moved and percent shown")

    # The research knobs: log axis, markers, combined files with a band.
    page.get_by_key("log_y").set_value(True).run()
    page.get_by_key("markers").set_value(True).run()
    page.get_by_key("combine").set_value(True).run()
    if page.exception:
        raise AssertionError(f"combining raised: {page.exception[0].value}")
    captions = [caption.value for caption in page.caption]
    print(f"    combined: {[text for text in captions if 'band' in text]}")
    assert any("band =" in text for text in captions), \
        "a combined chart must say what its band is"

    # And the figure files themselves, through the button on the page.
    page.get_by_key("image_format").set_value("svg").run()
    page.get_by_key("render_images").click().run()
    if page.exception:
        raise AssertionError(f"making the files raised: "
                             f"{page.exception[0].value}")
    offered = [button.label for button in page.download_button]
    print(f"    downloads offered: {offered}")
    assert any(label.endswith(".svg") for label in offered), \
        "no image file came out of the Make button"
    os.environ.pop(csvexplorer.PATHS_ENV, None)


def test_dashboard_serves_on_the_port_asked_for(folder):
    """`python csvexplorer.py <folder>` really starts Streamlit, and it answers.

    The case before this one draws the page inside this very process, so it
    uses no port at all. This one is the real thing: another process, a
    Streamlit server, a page a browser could open.
    """
    print("\n=== 13. the real server ===")
    port = free_port()
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    print(f"    python csvexplorer.py <folder>, on port {port}")

    launcher = subprocess.Popen(
        [sys.executable, MODULE_PATH, folder],
        # The port, and headless so no browser window opens on this machine.
        env=dict(os.environ, STREAMLIT_SERVER_PORT=str(port),
                 STREAMLIT_SERVER_HEADLESS="true",
                 STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        health = None
        for _ in range(START_SECONDS):          # a server takes a moment
            time.sleep(1)
            health = read_url(health_url, timeout=2)
            if health or launcher.poll() is not None:   # up, or gave up
                break
        if not health:
            raise AssertionError(
                f"nothing answered on port {port} within {START_SECONDS}s; "
                f"the launcher said:\n{launcher.communicate(timeout=10)[0]}")

        page = read_url(f"http://127.0.0.1:{port}", timeout=10)
        print(f"    health {health[1].strip()!r}, "
              f"page {page[0]}, {len(page[1]):,} bytes")
        assert health[1].strip() == "ok", f"the health check said {health[1]!r}"
        assert page[0] == 200 and page[1], f"the page answered {page}"
    finally:
        stop_process_tree(launcher)

    time.sleep(1)
    assert read_url(health_url, timeout=2) is None, \
        f"something is still listening on port {port}"
    print(f"    stopped, port {port} free again")


def main():
    folder = tempfile.mkdtemp(prefix="csvexplorer-test-")
    print(f"writing CSVs to {folder}")
    try:
        write_test_csvs(folder)

        test_finds_the_files(folder)
        test_one_table_for_every_file(folder)    # opens the CSVs the next
        test_chart_data_smoothing_and_thinning()  # four cases read
        test_summary_of_each_source()
        test_two_sources_side_by_side()
        test_one_column_per_source_is_chart_shaped()
        test_csv_that_names_its_own_sources()
        test_same_file_name_in_two_folders()
        load_csvs(folder)                        # the two sensor logs again
        test_files_of_one_setting_become_one_line()
        load_csvs(folder)
        test_chart_and_the_files_it_saves()
        test_dashboard_draws(folder)
        test_two_files_in_one_diagram(folder)
        test_dashboard_serves_on_the_port_asked_for(folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    print("\nall good\n")


if __name__ == "__main__":
    main()
