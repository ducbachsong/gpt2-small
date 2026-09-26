"""csvexplorer — CSV results loaded into DuckDB, compared and plotted for a paper.

    python csvexplorer.py results/*.csv     # opens the page in your browser

    from csvexplorer import load_csvs, plot_data, line_chart, chart_image
    load_csvs("results/*.csv")              # every CSV as one DuckDB table: data
    chart = line_chart(plot_data("step", ["loss"]))
    open("figure.pdf", "wb").write(chart_image(chart, "pdf"))

Nothing here knows what your CSVs hold: the columns come from the files, any
column can be the x axis, and any numeric column can be plotted. One CSV is
one source, named after the file, so two files are compared by plotting them
together; a CSV with a `source` column of its own keeps it, so one file can
hold many. Repeated runs of one setting can be aggregated into a single line
with a band showing their spread.

Charts are Altair, so they zoom, carry tooltips, take log axes, and export as
PNG, SVG or PDF at the resolution a journal asks for. Settings are at the top
of this file.
"""
import glob
import os
import re
import subprocess
import sys
import tempfile

__all__ = ["find_csv_files", "load_csvs", "current_connection",
           "use_connection", "run_sql", "list_sources", "list_columns",
           "numeric_columns", "default_x_column", "plot_data", "group_names",
           "aggregate_data", "label_lines", "rescale_lines", "line_chart",
           "chart_image", "wide_by_source", "wide_by_line", "summary_table",
           "compare_two_sources", "show_dashboard", "open_dashboard"]

# ── the settings ────────────────────────────────────────────────────────────
# Edit these for good. Most are also arguments of the functions below, which
# changes them for that one call and leaves the global alone.
CSV_PATHS = "*.csv"             # the files to read when none are given: a
                                # file, a folder, a glob, or several of those
                                # separated by ";"
TABLE_NAME = "data"             # the DuckDB table every function reads
SOURCE_COLUMN = "source"        # tells which file a row came from; taken from
                                # your CSV when it has a column of that name
ROW_COLUMN = "row_number"       # 1, 2, 3... down the file, added for CSVs
                                # with nothing else to line the rows up by
X_COLUMN_NAMES = ("time", "timestamp", "date", "datetime", "step", "epoch",
                  "iteration", "index")  # the first of these your CSVs have
                                # is the x axis until you pick another
SMOOTHING_POINTS = 0            # rolling mean over this many points; 0 is off
MAX_PLOT_POINTS = 4000          # a longer source is thinned out for the
                                # plot, so a million-row CSV still draws at
                                # once; the summary still reads every row
LOAD_INTO_MEMORY = True         # copy the CSVs into RAM once, so every query
                                # is instant; False re-reads them each query

# ── how the lines are drawn ─────────────────────────────────────────────────
LINE_SEPARATOR = " · "          # between the file and the column when both
                                # are drawn in one chart: "after · humidity"
GROUP_PATTERN = r"^(.*?)[-_ ]*\d*$"     # which files belong together when
                                # they are aggregated: the first bracket is
                                # the name kept, so "run-1" and "run-2" both
                                # become "run"
BAND = "standard deviation"     # what the shaded band shows around an
                                # aggregated line; see BANDS
BANDS = ("standard deviation", "standard error", "95% interval", "min to max",
         "quartiles", "none")
RESCALINGS = ("as they are", "minus first value", "percent from first")
CHART_KINDS = ("line", "step", "area", "scatter")
PALETTES = {
    # Okabe-Ito: made to stay apart for colour-blind readers and in print.
    "colour blind safe": ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                          "#CC79A7", "#56B4E9", "#F0E442", "#000000"],
    "tableau": ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
                "#EDC948", "#B07AA1", "#FF9DA7"],
    "greys (for print)": ["#000000", "#666666", "#999999", "#BBBBBB"],
}
PALETTE = "colour blind safe"
CHART_WIDTH = 760               # pixels; also the width of an exported image
CHART_HEIGHT = 380
IMAGE_FORMAT = "png"            # what the export buttons make: png, svg, pdf
IMAGE_DPI = 300                 # pixels per inch of an exported png; 300 is
                                # the usual least a journal takes, 600 for
                                # line art

PAGE_TITLE = "CSV Explorer"
PATHS_ENV = "CSVEXPLORER_PATHS"  # how open_dashboard() tells the Streamlit
                                 # page which files to read

_connection = None              # the DuckDB connection load_csvs() left behind


# ── finding and reading the files ───────────────────────────────────────────
def find_csv_files(paths=None):
    """Find the CSV files that `paths` names.

    In : paths  str | list[str] | None   e.g.
                  "results/before.csv"   one file
                  "results"              a folder: every .csv in it, and below
                  "results/*.csv"        a glob
                  "a.csv;b.csv"          several in one string, split on ";"
                  ["a.csv", "results"]   several in a list
                  None                   the CSV_PATHS setting
    Out: list[str] sorted absolute paths, e.g.
           ["C:/lab/results/after.csv", "C:/lab/results/before.csv"]
         [] when nothing matches.
    """
    if paths is None:
        paths = CSV_PATHS
    if isinstance(paths, str):
        entries = paths.split(";")
    else:
        entries = [str(entry) for entry in paths]

    found = []
    for entry in entries:
        entry = os.path.expanduser(entry.strip())
        if not entry:
            continue
        if os.path.isdir(entry):
            # A folder means every CSV in it, however deep.
            found.extend(glob.glob(os.path.join(entry, "**", "*.csv"),
                                   recursive=True))
        else:
            # A plain path comes back from glob when it exists, and a pattern
            # comes back expanded, so one call covers both.
            found.extend(glob.glob(entry, recursive=True))

    return sorted({os.path.abspath(path) for path in found})


def load_csvs(paths=None, load_into_memory=LOAD_INTO_MEMORY):
    """Read every CSV into one DuckDB table called `data`, and remember it.

    Every later call here uses this table until the next load_csvs(), so call
    it again to pick up rows that have been appended since.

    In : paths             as find_csv_files() takes them
         load_into_memory  bool  e.g. True, a table in RAM; False leaves a
                           view that re-reads the CSVs on every query
    Out: the duckdb connection, for SQL of your own. Raises FileNotFoundError
         when no CSV matches, and whatever DuckDB raises for a file it cannot
         read.
    """
    import duckdb

    csv_files = find_csv_files(paths)
    if not csv_files:
        raise FileNotFoundError(f"[csvexplorer] no CSV files matched {paths!r}")
    source_names = _source_names_for(csv_files)

    connection = duckdb.connect()
    selects = []
    for csv_file, source_name in zip(csv_files, source_names):
        # A CSV that already names its sources keeps its own column, so one
        # file can hold several; otherwise the file name becomes the name.
        csv_columns = _csv_column_names(connection, csv_file)
        added = []
        if SOURCE_COLUMN not in csv_columns:
            added.append(f"{quote_text(source_name)} AS {quote_name(SOURCE_COLUMN)}")
        if ROW_COLUMN not in csv_columns:
            added.append(f"row_number() OVER () AS {quote_name(ROW_COLUMN)}")
        selects.append(f"SELECT {', '.join(added + ['*'])} FROM "
                       f"read_csv_auto({quote_text(csv_file)}, sample_size = -1)")

    # BY NAME lines the files up by column name, so a file with one extra
    # column still fits; the others get NULL there.
    # sample_size = -1 above reads the whole file before deciding the types,
    # so a column that starts out looking like an integer is not cut short.
    table_or_view = "TABLE" if load_into_memory else "VIEW"
    connection.execute(f"CREATE OR REPLACE {table_or_view} "
                       f"{quote_name(TABLE_NAME)} AS\n"
                       + "\nUNION ALL BY NAME\n".join(selects))

    use_connection(connection)

    row_count = connection.execute(
        f"SELECT count(*) FROM {quote_name(TABLE_NAME)}").fetchone()[0]
    print(f"[csvexplorer] {len(csv_files)} file(s), {row_count:,} rows: "
          f"{', '.join(list_sources(connection))}")
    return connection


def current_connection():
    """The connection from the last load_csvs() or use_connection().

    In : nothing.
    Out: the duckdb connection. Raises ValueError when no CSVs are loaded yet.
    """
    if _connection is None:
        raise ValueError("[csvexplorer] nothing loaded yet: call "
                         "load_csvs(paths) first")
    return _connection


def use_connection(connection):
    """Make `connection` the one every function here reads.

    load_csvs() calls this itself. Call it by hand to hand the functions a
    connection they did not open — a DuckDB file of your own, or the one a
    Streamlit page kept from its last run.

    In : connection  a duckdb connection holding a `data` table
    Out: that same connection
    """
    global _connection
    _connection = connection
    return connection


def run_sql(sql, connection=None):
    """Run any SQL against the table and give back a DataFrame.

    In : sql         str   e.g. "SELECT source, max(accuracy) FROM data
                                 GROUP BY source"
         connection  a duckdb connection; None uses the last load_csvs()
    Out: pandas.DataFrame of the result
    """
    if connection is None:
        connection = current_connection()
    return connection.execute(sql).df()


# ── what is in the files ────────────────────────────────────────────────────
def list_sources(connection=None):
    """The source names in the table, one per file unless a CSV named its own.

    In : connection  None uses the last load_csvs()
    Out: list[str] sorted, e.g. ["after", "before"]
    """
    frame = run_sql(f"SELECT DISTINCT {quote_name(SOURCE_COLUMN)} AS source "
                    f"FROM {quote_name(TABLE_NAME)} ORDER BY 1", connection)
    return [str(name) for name in frame["source"]]


def list_columns(connection=None):
    """Every column in the table, with its type.

    In : connection  None uses the last load_csvs()
    Out: dict[str, str] in table order, e.g.
           {"source": "VARCHAR", "row_number": "BIGINT", "step": "BIGINT",
            "loss": "DOUBLE"}
    """
    if connection is None:
        connection = current_connection()
    described = connection.execute(f"DESCRIBE {quote_name(TABLE_NAME)}").fetchall()
    return {row[0]: row[1] for row in described}


def numeric_columns(connection=None):
    """The columns that hold numbers, the ones worth plotting.

    In : connection  None uses the last load_csvs()
    Out: list[str] e.g. ["row_number", "step", "loss", "accuracy"]
    """
    number_types = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
                    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "FLOAT",
                    "DOUBLE", "REAL", "DECIMAL", "NUMERIC")
    chosen = []
    for name, column_type in list_columns(connection).items():
        if column_type.upper().startswith(number_types):
            chosen.append(name)
    return chosen


def default_x_column(connection=None):
    """The column to put on the x axis until you pick another.

    In : connection  None uses the last load_csvs()
    Out: str   the first of X_COLUMN_NAMES your CSVs have, e.g. "step";
               ROW_COLUMN when they have none of them
    """
    columns = list_columns(connection)
    for name in X_COLUMN_NAMES:
        if name in columns:
            return name
    return ROW_COLUMN


# ── the numbers to plot ─────────────────────────────────────────────────────
def plot_data(x_column=None, value_columns=None, sources=None,
              smoothing=SMOOTHING_POINTS, max_points=MAX_PLOT_POINTS,
              connection=None):
    """The values of every chosen column, source by source, ready to plot.

    In : x_column       str | None   e.g. "step"; None takes default_x_column()
         value_columns  list[str] | str | None   e.g. ["loss"];
                        None takes every numeric column but the x axis
         sources        list[str] | str | None   e.g. ["before", "after"];
                        None takes every source
         smoothing      int   e.g. 20, a rolling mean over 20 points; 0 is off
         max_points     int   e.g. 4000, the most points kept per source and
                        column; a longer one is thinned out evenly
         connection     None uses the last load_csvs()
    Out: pandas.DataFrame, one row per point, e.g.
              source  value_column  x   value
           0  after   loss          1   2.31
           1  after   loss          2   2.04
           ...
         Sorted by value_column, source, x. Empty when nothing matches.
    """
    picked = _picked_sql(x_column, value_columns, sources, connection)
    return run_sql(f"""
        WITH picked AS ({picked}),
        numbered AS (
            SELECT *,
                   row_number() OVER (PARTITION BY source, value_column
                                      ORDER BY x) AS point_number,
                   count(*) OVER (PARTITION BY source, value_column)
                       AS point_count
            FROM picked
        ),
        smoothed AS (
            SELECT source, value_column, x, point_number, point_count,
                   avg(value) OVER w AS value
            FROM numbered
            WINDOW w AS (PARTITION BY source, value_column ORDER BY x
                         {_window_sql(smoothing)})
        )
        SELECT source, value_column, x, value
        FROM smoothed
        -- Keep every nth point, and the last one, so the line still ends
        -- where the data ended.
        WHERE point_number % {_keep_every_sql(max_points)} = 0
           OR point_number = point_count
        ORDER BY value_column, source, x
    """, connection)


def group_names(sources, pattern=GROUP_PATTERN):
    """Work out which files belong together, from their names.

    The same setting run several times — different seeds, different days —
    is usually named for the setting with something tacked on. The default
    pattern drops a trailing number, so those land in one group and can be
    plotted as one line with a band.

    In : sources  list[str]  e.g. ["run-1", "run-2", "tuned-1"]
         pattern  str   a regular expression whose first bracket is the name
                  to keep, e.g. r"^(.*?)[-_ ]*\\d*$"
    Out: dict[str, str] in the order given, e.g.
           {"run-1": "run", "run-2": "run", "tuned-1": "tuned"}
         A name the pattern does not match keeps itself.
    """
    groups = {}
    for source in sources:
        found = re.match(pattern, source)
        name = found.group(1) if found and found.lastindex else ""
        groups[source] = name or source
    return groups


def aggregate_data(x_column=None, value_columns=None, sources=None,
                   pattern=GROUP_PATTERN, band=BAND,
                   smoothing=SMOOTHING_POINTS, max_points=MAX_PLOT_POINTS,
                   connection=None):
    """Aggregate the files of each group into one line, and say how they spread.

    The line is the mean of the files at each x; the band around it is what
    `band` asks for. This is the usual way to report several runs of one
    setting in a paper — one line per setting, not one per run.

    In : x_column, value_columns, sources, smoothing, max_points
                   as plot_data() takes them
         pattern   str   how names are grouped; see group_names()
         band      str   one of BANDS:
                     "standard deviation"  mean ± sd, how far the files spread
                     "standard error"      mean ± sd/sqrt(files)
                     "95% interval"        mean ± 1.96 sd/sqrt(files)
                     "min to max"          every file inside the band
                     "quartiles"           the middle half of the files
                     "none"                no band, just the mean
    Out: pandas.DataFrame, one row per group, column and x, e.g.
              group_name  value_column  x   value   low   high  files
           0  run         loss          1   2.04    1.98  2.10      3
         `files` is how many files went into that point, worth stating in a
         caption. Without a band the low and high columns are left out.
    """
    if sources is None:
        sources = list_sources(connection)
    elif isinstance(sources, str):
        sources = [sources]
    if not sources:
        raise ValueError("[csvexplorer] no sources to aggregate")
    if band not in BANDS:
        raise ValueError(f"[csvexplorer] band must be one of {BANDS}")

    groups = group_names(sources, pattern)
    pairs = ", ".join(f"({quote_text(source)}, {quote_text(group_name)})"
                      for source, group_name in groups.items())
    low, high = {
        "standard deviation": ("value - spread", "value + spread"),
        "standard error": ("value - spread / sqrt(files)",
                           "value + spread / sqrt(files)"),
        "95% interval": ("value - 1.96 * spread / sqrt(files)",
                         "value + 1.96 * spread / sqrt(files)"),
        "min to max": ("lowest", "highest"),
        "quartiles": ("lower_quarter", "upper_quarter"),
        "none": ("value", "value"),
    }[band]

    picked = _picked_sql(x_column, value_columns, sources, connection)
    frame = run_sql(f"""
        WITH picked AS ({picked}),
        named AS (
            SELECT names.group_name AS group_name,
                   picked.value_column, picked.x, picked.value
            FROM picked
            JOIN (VALUES {pairs}) AS names(source, group_name)
              ON picked.source = names.source
        ),
        grouped AS (
            SELECT group_name, value_column, x,
                   avg(value)                       AS value,
                   count(*)                         AS files,
                   -- one file on its own spreads by nothing, not by NULL
                   coalesce(stddev_samp(value), 0)  AS spread,
                   min(value)                       AS lowest,
                   max(value)                       AS highest,
                   quantile_cont(value, 0.25)       AS lower_quarter,
                   quantile_cont(value, 0.75)       AS upper_quarter
            FROM named
            GROUP BY group_name, value_column, x
        ),
        edged AS (
            SELECT group_name, value_column, x, value, files,
                   {low} AS low, {high} AS high
            FROM grouped
        ),
        numbered AS (
            SELECT *,
                   row_number() OVER (PARTITION BY group_name, value_column
                                      ORDER BY x) AS point_number,
                   count(*) OVER (PARTITION BY group_name, value_column)
                       AS point_count
            FROM edged
        ),
        smoothed AS (
            SELECT group_name, value_column, x, files,
                   point_number, point_count,
                   avg(value) OVER w AS value,
                   avg(low) OVER w   AS low,
                   avg(high) OVER w  AS high
            FROM numbered
            WINDOW w AS (PARTITION BY group_name, value_column ORDER BY x
                         {_window_sql(smoothing)})
        )
        SELECT group_name, value_column, x, value, low, high, files
        FROM smoothed
        WHERE point_number % {_keep_every_sql(max_points)} = 0
           OR point_number = point_count
        ORDER BY value_column, group_name, x
    """, connection)

    if band == "none":
        frame = frame.drop(columns=["low", "high"])
    return frame


def label_lines(frame, separator=LINE_SEPARATOR):
    """Add the `line` column, the name each plotted line goes by.

    In : frame      from plot_data() or aggregate_data()
         separator  str   between the two names, e.g. " · "
    Out: the same frame with a `line` column: the source (or group) on its
         own while one value column is plotted, "after · loss" when more than
         one is, so a legend says which is which.
    """
    if "line" in frame.columns:
        return frame
    name_column = "group_name" if "group_name" in frame.columns else "source"
    if frame.empty:
        return frame.assign(line=[])
    if frame["value_column"].nunique() > 1:
        line = frame[name_column] + separator + frame["value_column"]
    else:
        line = frame[name_column]
    return frame.assign(line=line)


def rescale_lines(frame, how="as they are"):
    """Put lines of very different sizes on one scale.

    In : frame  from plot_data() or aggregate_data(), with a `line` column
         how    str   one of RESCALINGS:
                  "as they are"        untouched
                  "minus first value"  every line starts at 0, shapes compare
                  "percent from first" every line in percent of where it
                                       started, so 2.3 and 20.4 compare
    Out: the same frame with value — and low and high when it has them —
         rescaled. A line starting at 0 comes back as inf in percent.
    """
    if how == "as they are" or frame.empty:
        return frame
    if how not in RESCALINGS:
        raise ValueError(f"[csvexplorer] how must be one of {RESCALINGS}")

    frame = label_lines(frame).sort_values(["line", "x"])
    first_values = frame.groupby("line")["value"].transform("first")
    rescaled = frame.copy()
    for column in ("value", "low", "high"):
        if column in frame.columns:
            if how == "minus first value":
                rescaled[column] = frame[column] - first_values
            else:
                rescaled[column] = (frame[column] / first_values - 1.0) * 100.0
    return rescaled


def wide_by_source(frame, value_column=None):
    """Turn plot_data()'s rows into one column per source, for plain pandas.

    line_chart() does not need this; DataFrame.plot() and st.line_chart() do.

    In : frame         the DataFrame from plot_data()
         value_column  str | None   which value column to take, e.g. "loss";
                       None when the frame holds only one
    Out: pandas.DataFrame indexed by x, one column per source, e.g.
           x    after  before
           1     2.28    2.31
           2     1.97    2.04
    """
    if value_column is not None:
        frame = frame[frame["value_column"] == value_column]
    # pivot_table, not pivot: a source with the same x twice would otherwise
    # raise, and the mean of the two is the honest answer.
    return frame.pivot_table(index="x", columns="source", values="value",
                             aggfunc="mean")


def wide_by_line(frame, separator=LINE_SEPARATOR):
    """Turn plot_data()'s rows into one column per file-and-column pair.

    In : frame      the DataFrame from plot_data()
         separator  str   between the two names, e.g. " · "
    Out: pandas.DataFrame indexed by x, one column per line, e.g.
           x   after · loss  before · loss  after · accuracy
           1           2.28           2.31              0.12
    """
    labelled = frame.assign(line=frame["source"] + separator
                            + frame["value_column"])
    return labelled.pivot_table(index="x", columns="line", values="value",
                                aggfunc="mean")


def summary_table(x_column=None, value_columns=None, sources=None,
                  connection=None):
    """Where each source started, ended, and how far it went in between.

    In : as plot_data(), without the smoothing and thinning
    Out: pandas.DataFrame, one row per source and value column, e.g.
              source  value_column  points  first_value  last_value  smallest  largest  mean_value
           0  after   loss            1000         2.28        0.35      0.34     2.28        0.79
           1  before  loss            1000         2.31        0.42      0.41     2.31        0.88
         first_value and last_value follow the x axis, not the file order.
    """
    picked = _picked_sql(x_column, value_columns, sources, connection)
    return run_sql(f"""
        WITH picked AS ({picked})
        SELECT source,
               value_column,
               count(*)             AS points,
               arg_min(value, x)    AS first_value,
               arg_max(value, x)    AS last_value,
               min(value)           AS smallest,
               max(value)           AS largest,
               avg(value)           AS mean_value
        FROM picked
        GROUP BY source, value_column
        ORDER BY value_column, source
    """, connection)


def compare_two_sources(source_a, source_b, value_column, x_column=None,
                        connection=None):
    """Put two sources side by side, and take one away from the other.

    The two need not have the same x values: each point of source_a is
    matched with the last point of source_b at or before its x.

    In : source_a      str   e.g. "after", the one the difference is about
         source_b      str   e.g. "before", the one taken away
         value_column  str   e.g. "loss"
         x_column      str | None   e.g. "step"; None takes default_x_column()
         connection    None uses the last load_csvs()
    Out: pandas.DataFrame, one row per point of source_a, e.g.
              x  after  before  difference
           0  1   2.28    2.31       -0.03
           1  2   1.97    2.04       -0.07
         The two middle columns are named after the sources.
    """
    picked = _picked_sql(x_column, [value_column], [source_a, source_b],
                         connection)
    return run_sql(f"""
        WITH picked AS ({picked}),
        source_a AS (SELECT x, value FROM picked
                     WHERE source = {quote_text(source_a)}),
        source_b AS (SELECT x, value FROM picked
                     WHERE source = {quote_text(source_b)})
        SELECT a.x                  AS x,
               a.value              AS {quote_name(source_a)},
               b.value              AS {quote_name(source_b)},
               a.value - b.value    AS difference
        FROM source_a AS a
        -- ASOF, not a plain join: two files rarely line up row for row.
        ASOF JOIN source_b AS b ON a.x >= b.x
        ORDER BY a.x
    """, connection)


# ── the chart ───────────────────────────────────────────────────────────────
def line_chart(frame, title="", x_title=None, y_title=None, kind="line",
               markers=False, log_x=False, log_y=False, width=CHART_WIDTH,
               height=CHART_HEIGHT, palette=PALETTE, band_opacity=0.2,
               separator=LINE_SEPARATOR):
    """A 2D chart of every line in `frame` — the one chart this tool draws.

    In : frame         long, from plot_data() or aggregate_data(); a `line`
                       column is added when it has none, and low and high
                       columns, when it has them, become a shaded band
         title         str   over the chart, e.g. "validation loss"
         x_title       str | None   the x axis label; None leaves "x"
         y_title       str | None   the y axis label; None leaves "value"
         kind          str   one of CHART_KINDS: line, step, area, scatter
         markers       bool  a marker on every point, for few points or print
         log_x, log_y  bool  log axes; values must be above zero
         width, height int   pixels, and the size of an exported image
         palette       str | list[str]   a name from PALETTES, or your own
                       colours; the default stays apart for colour-blind
                       readers and in greyscale print
         band_opacity  float e.g. 0.2, how solid the band around a line is
    Out: an altair.Chart. Draw it with st.altair_chart(chart), export it with
         chart_image(chart, "pdf"), or keep the plain Vega-Lite behind it
         with chart.to_json(), which is worth archiving beside a paper.
    """
    import altair as alt
    import pandas

    frame = label_lines(frame, separator)
    if isinstance(palette, str):
        colours = PALETTES.get(palette, PALETTES[PALETTE])
    else:
        colours = list(palette)

    if pandas.api.types.is_numeric_dtype(frame["x"]):
        x_kind = "quantitative"
    elif pandas.api.types.is_datetime64_any_dtype(frame["x"]):
        x_kind = "temporal"
    else:
        x_kind = "nominal"

    x_axis = alt.X(f"x:{x_kind[0].upper()}", title=x_title,
                   scale=alt.Scale(type="log" if log_x and x_kind == "quantitative"
                                   else "linear", zero=False))
    y_scale = alt.Scale(type="log" if log_y else "linear", zero=False)
    y_axis = alt.Y("value:Q", title=y_title, scale=y_scale)
    colour = alt.Color("line:N", title=None,
                       scale=alt.Scale(range=list(colours)))
    hover = [alt.Tooltip("line:N", title="line"),
             alt.Tooltip(f"x:{x_kind[0].upper()}", title=x_title or "x"),
             alt.Tooltip("value:Q", title=y_title or "value", format=".6g")]

    base = alt.Chart(frame)
    layers = []
    if {"low", "high"} <= set(frame.columns):
        # The band goes down first, so the lines stay readable on top of it.
        layers.append(base.mark_area(opacity=band_opacity).encode(
            x=x_axis,
            y=alt.Y("low:Q", title=y_title, scale=y_scale),
            y2=alt.Y2("high:Q"),
            color=colour))

    if kind == "area":
        marks = base.mark_area(opacity=0.4, line=True)
    elif kind == "scatter":
        marks = base.mark_point(filled=True, size=30)
    elif kind == "step":
        marks = base.mark_line(interpolate="step-after", point=markers)
    else:
        marks = base.mark_line(point=markers)
    layers.append(marks.encode(x=x_axis, y=y_axis, color=colour, tooltip=hover))

    return (alt.layer(*layers)
            .properties(width=width, height=height, title=title)
            .interactive())     # drag to pan, wheel to zoom


def chart_image(chart, image_format=IMAGE_FORMAT, dpi=IMAGE_DPI):
    """The chart as a file: the figure to put in a paper.

    In : chart         from line_chart()
         image_format  "png" | "svg" | "pdf"; svg and pdf are vector, so they
                       stay sharp at any size a journal prints them
         dpi           int   pixels per inch of a png, e.g. 300
    Out: bytes   the file's contents, ready for open(..., "wb").write() or
         Streamlit's download button. Raises ValueError for another format,
         and RuntimeError when vl-convert-python is not installed:
         pip install vl-convert-python
    """
    import io

    if image_format not in ("png", "svg", "pdf"):
        raise ValueError("[csvexplorer] image_format must be png, svg or pdf")
    try:
        if image_format == "svg":
            text = io.StringIO()
            chart.save(text, format="svg")
            return text.getvalue().encode("utf-8")
        binary = io.BytesIO()
        if image_format == "png":
            chart.save(binary, format="png", ppi=dpi)
        else:
            chart.save(binary, format="pdf")
        return binary.getvalue()
    except ImportError as error:
        raise RuntimeError("[csvexplorer] exporting images needs vl-convert: "
                           "pip install vl-convert-python") from error


# ── the SQL behind them ─────────────────────────────────────────────────────
def _picked_sql(x_column, value_columns, sources, connection=None):
    """The SELECT that turns the chosen columns into (source, value_column, x, value).

    One row per point per value column, which is the shape every function
    above works from.

    In : x_column, value_columns, sources   as plot_data() takes them
    Out: str   a SELECT, ready to use as a CTE
    """
    if x_column is None:
        x_column = default_x_column(connection)
    if value_columns is None:
        value_columns = [name for name in numeric_columns(connection)
                         if name not in (x_column, ROW_COLUMN)]
    elif isinstance(value_columns, str):
        value_columns = [value_columns]
    if not value_columns:
        raise ValueError("[csvexplorer] no value columns to plot: the CSVs "
                         "hold no numbers besides the x axis")

    if sources is None:
        source_filter = ""
    else:
        if isinstance(sources, str):
            sources = [sources]
        wanted = ", ".join(quote_text(name) for name in sources)
        source_filter = f"AND {quote_name(SOURCE_COLUMN)} IN ({wanted})"

    selects = []
    for value_column in value_columns:
        selects.append(f"""
            SELECT {quote_name(SOURCE_COLUMN)}       AS source,
                   {quote_text(value_column)}        AS value_column,
                   {quote_name(x_column)}            AS x,
                   CAST({quote_name(value_column)} AS DOUBLE) AS value
            FROM {quote_name(TABLE_NAME)}
            WHERE {quote_name(x_column)} IS NOT NULL
              AND {quote_name(value_column)} IS NOT NULL
              {source_filter}""")
    return "\nUNION ALL\n".join(selects)


def _window_sql(smoothing):
    """The window frame a rolling mean of `smoothing` points needs.

    In : smoothing  int   e.g. 20; 0 and 1 both mean no smoothing
    Out: str   e.g. "ROWS BETWEEN 19 PRECEDING AND CURRENT ROW"
    """
    return (f"ROWS BETWEEN {max(int(smoothing) - 1, 0)} PRECEDING "
            f"AND CURRENT ROW")


def _keep_every_sql(max_points):
    """How many points to step over to come out at `max_points` of them.

    In : max_points  int   e.g. 4000
    Out: str   an expression over point_count, e.g.
               "greatest(1, CAST(ceil(point_count / 4000.0) AS BIGINT))"
    """
    return (f"greatest(1, CAST(ceil(point_count / {float(max_points)}) "
            f"AS BIGINT))")


def quote_name(name):
    """Wrap a table or column name for SQL, whatever is in it.

    In : name  str   e.g. 'train loss'
    Out: str         e.g. '"train loss"'
    """
    return '"' + str(name).replace('"', '""') + '"'


def quote_text(value):
    """Wrap a value for SQL as text.

    In : value  str   e.g. "o'clock"
    Out: str          e.g. "'o''clock'"
    """
    return "'" + str(value).replace("'", "''") + "'"


def _source_names_for(csv_files):
    """Name each file, keeping the names apart when two files share one.

    In : csv_files  list[str]  e.g. ["/lab/a/log.csv", "/lab/b/log.csv"]
    Out: list[str] in the same order, e.g. ["a/log", "b/log"]
    """
    plain_names = [os.path.splitext(os.path.basename(path))[0]
                   for path in csv_files]
    names = []
    for path, plain_name in zip(csv_files, plain_names):
        if plain_names.count(plain_name) > 1:
            # Two files called the same thing: the folder tells them apart.
            folder = os.path.basename(os.path.dirname(path))
            name = f"{folder}/{plain_name}"
        else:
            name = plain_name
        while name in names:            # still the same: mark it
            name += "+"
        names.append(name)
    return names


def _csv_column_names(connection, csv_file):
    """The column names of one CSV, without reading all of it.

    In : connection  a duckdb connection
         csv_file    str   e.g. "C:/lab/results/before.csv"
    Out: list[str]   e.g. ["step", "loss"]
    """
    described = connection.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto({quote_text(csv_file)})").fetchall()
    return [row[0] for row in described]


# ── the page ────────────────────────────────────────────────────────────────
def show_dashboard(paths=None):
    """Draw the Streamlit page: charts, a summary, and two sources compared.

    Meant to be run by open_dashboard() or `streamlit run csvexplorer.py`,
    not called by hand.

    In : paths  as find_csv_files() takes them; None takes the CSV_PATHS
                setting
    Out: None. Draws the page and returns when Streamlit has finished this
         run of the script.
    """
    import streamlit as st

    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    _load_from_sidebar(st, paths)
    choices = _choices_from_sidebar(st)
    if not choices["sources"] or not choices["value_columns"]:
        st.info("Select at least one source and one value in the sidebar.")
        st.stop()

    charts, frame = _show_charts(st, choices)
    _show_export(st, charts, frame, choices)
    _show_summary(st, choices)
    _show_rows_and_sql(st, frame)


def _pick_many(st, label, options, key, help_text=None):
    """A multiselect built to cope with a long list of options.

    Above the list sit a filter box and three buttons, so a hundred files or
    columns can be narrowed to the handful wanted without scrolling: type a
    few letters, then Select these, Add these, or Clear.

    In : st         the streamlit module, or a sidebar or column of it
         label      str   e.g. "Sources"
         options    list[str]  everything on offer
         key        str   the session key, e.g. "sources"
         help_text  str | None
    Out: list[str]   what is selected now
    """
    options = list(options)
    # Whatever was selected before may be gone after a reload, so trim it to
    # what exists now: Streamlit refuses a selection it cannot offer.
    st.session_state[key] = [name for name in st.session_state.get(key, options)
                             if name in options]

    filter_text = st.text_input(f"Filter {label.lower()}", key=f"{key}_filter",
                                placeholder="type to narrow the list")
    matching = [name for name in options
                if filter_text.lower() in name.lower()] if filter_text else options

    select_these, add_these, clear = st.columns(3)
    if select_these.button("Only these", key=f"{key}_only",
                           disabled=not filter_text,
                           help="select what matches the filter"):
        st.session_state[key] = matching
    if add_these.button("Add these", key=f"{key}_add", disabled=not filter_text,
                        help="add what matches to the selection"):
        st.session_state[key] = st.session_state[key] + [
            name for name in matching if name not in st.session_state[key]]
    if clear.button("Clear", key=f"{key}_clear"):
        st.session_state[key] = []

    picked = st.multiselect(label, options, key=key, help=help_text)
    st.caption(f"{len(picked)} of {len(options)} selected"
               + (f", {len(matching)} match the filter" if filter_text else ""))
    return picked


def _load_from_sidebar(st, paths):
    """The Files section: where the CSVs come from, and which of them to use."""
    with st.sidebar.expander("Files", expanded=True):
        typed_paths = st.text_area(
            "Paths (one per line)", key="paths", height=90,
            value=_paths_text(paths).replace(";", "\n"),
            help="a file, a folder, or a pattern such as results/*.csv")
        uploads = st.file_uploader("Or upload CSV files", type=["csv"],
                                   accept_multiple_files=True, key="uploads")

        found = find_csv_files([line for line in typed_paths.splitlines()])
        found = sorted(set(found) | set(_saved_uploads(st, uploads)))
        if not found:
            st.warning("No CSV files found at those paths.")
            st.stop()

        # Files are chosen by their source name, the same name the charts and
        # the table use, not by their whole path.
        by_name = dict(zip(_source_names_for(found), found))
        chosen_names = _pick_many(st, "Files", list(by_name), "files",
                                  help_text="which of the files found to load")
        chosen_paths = [by_name[name] for name in chosen_names]
        if not chosen_paths:
            st.warning("No files selected.")
            st.stop()
        reload_asked = st.button("Reload files", key="reload")

    # Streamlit runs this file again from the top on every click, so the
    # connection cannot live in a module variable: it is kept in the session
    # and handed back to the functions here each time.
    loaded_before = st.session_state.get("loaded_connection")
    if reload_asked or loaded_before is None or \
            st.session_state.get("loaded_paths") != chosen_paths:
        try:
            st.session_state["loaded_connection"] = load_csvs(chosen_paths)
        except Exception as error:
            st.sidebar.error(str(error))
            st.stop()
        st.session_state["loaded_paths"] = chosen_paths
    else:
        use_connection(loaded_before)


def _saved_uploads(st, uploads):
    """Keep uploaded CSVs on disk, where DuckDB can read them.

    In : st       the streamlit module
         uploads  what st.file_uploader() returned
    Out: list[str]  their paths, [] when nothing was uploaded
    """
    if not uploads:
        return []
    folder = st.session_state.get("upload_folder")
    if folder is None or not os.path.isdir(folder):
        folder = tempfile.mkdtemp(prefix="csvexplorer-uploads-")
        st.session_state["upload_folder"] = folder

    paths = []
    for upload in uploads:
        path = os.path.join(folder, upload.name)
        # Streamlit hands the same uploads back on every run; write once.
        if not os.path.exists(path) or os.path.getsize(path) != upload.size:
            with open(path, "wb") as saved_file:
                saved_file.write(upload.getbuffer())
        paths.append(path)
    return paths


def _choices_from_sidebar(st):
    """Every control in the sidebar, in one dict.

    In : st  the streamlit module
    Out: dict   what to plot and how, e.g.
           {"sources": ["before", "after"], "x_column": "step", ...}
    """
    # Every control has a key, so the page can find its own widgets again and
    # a test can reach them by name rather than by counting.
    all_sources = list_sources()
    all_columns = list(list_columns())
    x_default = default_x_column()

    with st.sidebar.expander("Series", expanded=True):
        chosen_sources = _pick_many(st, "Sources", all_sources, "sources",
                                    help_text="the files, or the runs inside "
                                              "them, to plot")
        x_column = st.selectbox("X axis", all_columns, key="x_column",
                                index=all_columns.index(x_default))
        value_choices = [name for name in numeric_columns()
                         if name not in (x_column, ROW_COLUMN)]
        if "values" not in st.session_state:
            st.session_state["values"] = value_choices[:4]
        chosen_values = _pick_many(st, "Values", value_choices, "values",
                                   help_text="the numeric columns to plot")

    one_chart = st.sidebar.radio(
        "Charts", ["one per value", "everything in one"], key="layout",
        help="everything in one puts any mix of files and columns in a "
             "single diagram") == "everything in one"
    smoothing = st.sidebar.slider("Smoothing (points)", 0, 100,
                                  SMOOTHING_POINTS, key="smoothing")

    with st.sidebar.expander("Repeated runs"):
        st.caption("Several runs of one setting: plot their mean, with a "
                   "band showing how far they spread.")
        combine = st.checkbox("Aggregate repeated runs", key="combine")
        pattern = st.text_input("Group names by", value=GROUP_PATTERN,
                                key="pattern",
                                help="a regular expression; the first bracket "
                                     "is the name kept, so run-1 and run-2 "
                                     "both become run")
        band = st.selectbox("Band", BANDS, index=BANDS.index(BAND), key="band")

    with st.sidebar.expander("Appearance"):
        chart_kind = st.selectbox("Chart type", CHART_KINDS, key="chart_kind")
        markers = st.checkbox("Show markers", key="markers")
        log_x = st.checkbox("Log x axis", key="log_x")
        log_y = st.checkbox("Log y axis", key="log_y")
        palette = st.selectbox("Colours", list(PALETTES), key="palette",
                               index=list(PALETTES).index(PALETTE))
        shown_as = st.selectbox("Value scaling", RESCALINGS, key="shown_as",
                                help="the last two put values of different "
                                     "sizes on one scale")
        width = st.number_input("Width (px)", 320, 2000, CHART_WIDTH, 20,
                                key="width")
        height = st.number_input("Height (px)", 200, 1600, CHART_HEIGHT, 20,
                                 key="height")
        title = st.text_input("Title", key="title")
        x_title = st.text_input("X label", value=x_column, key="x_title")
        y_title = st.text_input("Y label", key="y_title")

    with st.sidebar.expander("Export"):
        image_format = st.selectbox("Image format", ["png", "svg", "pdf"],
                                    key="image_format",
                                    help="svg and pdf are vector: sharp at "
                                         "any size a journal prints them")
        dpi = st.number_input("Resolution (dpi, png only)", 72, 1200,
                              IMAGE_DPI, 50, key="dpi")

    return {"sources": chosen_sources, "x_column": x_column,
            "value_columns": chosen_values, "one_chart": one_chart,
            "smoothing": smoothing, "combine": combine, "pattern": pattern,
            "band": band, "chart_kind": chart_kind, "markers": markers,
            "log_x": log_x, "log_y": log_y, "palette": palette,
            "shown_as": shown_as, "width": int(width), "height": int(height),
            "title": title, "x_title": x_title, "y_title": y_title,
            "image_format": image_format, "dpi": int(dpi)}


def _show_charts(st, choices):
    """Draw the charts the sidebar asks for.

    In : st, choices  from _choices_from_sidebar()
    Out: (charts, frame)
           charts  dict[str, altair.Chart]  by the name each is exported as
           frame   the points behind them
    """
    if choices["combine"]:
        frame = aggregate_data(choices["x_column"], choices["value_columns"],
                               choices["sources"], pattern=choices["pattern"],
                               band=choices["band"],
                               smoothing=choices["smoothing"])
    else:
        frame = plot_data(choices["x_column"], choices["value_columns"],
                          choices["sources"], smoothing=choices["smoothing"])
    frame = rescale_lines(label_lines(frame), choices["shown_as"])

    def draw(part, title, y_title):
        return line_chart(part, title=title, x_title=choices["x_title"],
                          y_title=y_title, kind=choices["chart_kind"],
                          markers=choices["markers"], log_x=choices["log_x"],
                          log_y=choices["log_y"], width=choices["width"],
                          height=choices["height"], palette=choices["palette"])

    charts = {}
    if choices["one_chart"]:
        st.subheader("All series")
        every_line = sorted(frame["line"].unique())
        # The picker sits by the chart, where the names it offers come from.
        chosen_lines = _pick_many(st, "Lines", every_line, "lines")
        kept = frame[frame["line"].isin(chosen_lines)]
        if kept.empty:
            st.info("Select at least one line.")
        else:
            charts["chart"] = draw(kept, choices["title"], choices["y_title"])
            st.altair_chart(charts["chart"])
    else:
        for value_column in choices["value_columns"]:
            part = frame[frame["value_column"] == value_column]
            if part.empty:
                continue
            st.subheader(value_column)
            charts[value_column] = draw(
                part, choices["title"], choices["y_title"] or value_column)
            st.altair_chart(charts[value_column])

    if choices["combine"] and "files" in frame.columns and not frame.empty:
        # State what the band is: a figure without that is not evidence.
        st.caption(f"line = mean of up to {int(frame['files'].max())} file(s); "
                   f"band = {choices['band']}")
    return charts, frame


def _show_export(st, charts, frame, choices):
    """The buttons that turn the charts on the page into files."""
    kinds = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}
    image_format = choices["image_format"]

    with st.expander("Export"):
        st.download_button("Plotted data (CSV)", frame.to_csv(index=False),
                           key="download_rows", file_name="csvexplorer.csv",
                           mime="text/csv")
        # Images are rendered only when asked for: every one costs a moment.
        if st.button(f"Render {image_format.upper()} files", key="render_images"):
            rendered = {}
            for name, chart in charts.items():
                try:
                    rendered[name] = chart_image(chart, image_format,
                                                 choices["dpi"])
                except Exception as error:
                    st.error(f"{name}: {error}")
            st.session_state["rendered_images"] = (image_format, rendered)

        rendered_format, rendered = st.session_state.get("rendered_images",
                                                         (None, {}))
        for name, image_bytes in rendered.items():
            st.download_button(f"{name}.{rendered_format}", image_bytes,
                               file_name=f"{name}.{rendered_format}",
                               mime=kinds.get(rendered_format,
                                              "application/octet-stream"),
                               key=f"download_{name}")


def _show_summary(st, choices):
    """The summary table, and two sources taken away from each other."""
    st.subheader("Summary")
    summary = summary_table(choices["x_column"], choices["value_columns"],
                            choices["sources"])
    st.dataframe(summary)

    if len(choices["sources"]) != 2:
        return
    import pandas

    source_a, source_b = choices["sources"]
    st.subheader(f"Difference: {source_a} minus {source_b}")
    differences = {}
    for value_column in choices["value_columns"]:
        comparison = compare_two_sources(source_a, source_b, value_column,
                                         choices["x_column"])
        differences[value_column] = comparison.set_index("x")["difference"]
    st.line_chart(pandas.DataFrame(differences))

    last_values = summary.pivot(index="value_column", columns="source",
                                values="last_value")
    last_values["difference"] = last_values[source_a] - last_values[source_b]
    st.caption("where each source ended")
    st.dataframe(last_values)


def _show_rows_and_sql(st, frame):
    """The points behind the charts, and a box for SQL of your own."""
    with st.expander("Rows"):
        st.dataframe(frame)

    with st.expander("SQL"):
        typed_sql = st.text_area("Your own query", key="sql",
                                 value=f"SELECT * FROM {TABLE_NAME} LIMIT 20",
                                 height=100)
        if st.button("Run query"):
            try:
                st.dataframe(run_sql(typed_sql))
            except Exception as error:
                st.error(str(error))


def open_dashboard(paths=None, port=None, wait=True):
    """Start Streamlit on this file, so the page opens in your browser.

    Blocks until you stop it with Ctrl+C, unless wait=False.

    In : paths  as find_csv_files() takes them; None takes the CSV_PATHS
                setting
         port   int | None   e.g. 8502; None lets Streamlit choose
         wait   bool         False starts it in the background and returns
                             at once, e.g. to watch a log while it is written
    Out: int    Streamlit's exit code, when wait=True
         subprocess.Popen   the running launcher, when wait=False; it keeps
                            running after your script ends
    """
    command = [sys.executable, "-m", "streamlit", "run",
               os.path.abspath(__file__)]
    if port is not None:
        command += ["--server.port", str(port)]

    # The page is this same file, run again by Streamlit; the environment
    # tells it which CSVs to read, and that it is the page rather than the
    # launcher.
    environment = dict(os.environ)
    environment[PATHS_ENV] = _paths_text(paths)
    print(f"[csvexplorer] streamlit run "
          f"{os.path.basename(__file__)} on {environment[PATHS_ENV]}")
    if not wait:
        return subprocess.Popen(command, env=environment)
    return subprocess.call(command, env=environment)


def _paths_text(paths):
    """The paths as one string, the shape the page and the environment use.

    In : paths  str | list[str] | None
    Out: str    e.g. "results/a.csv;results/b.csv"
    """
    if paths is None:
        return CSV_PATHS
    if isinstance(paths, str):
        return paths
    return ";".join(str(path) for path in paths)


def _running_under_streamlit():
    """Whether Streamlit is already running this file.

    In : nothing.
    Out: bool   True inside `streamlit run`, False under plain python
    """
    try:
        import streamlit.runtime
        return streamlit.runtime.exists()
    except Exception:
        return False


if __name__ == "__main__":
    if PATHS_ENV in os.environ or _running_under_streamlit():
        show_dashboard(os.environ.get(PATHS_ENV) or (sys.argv[1:] or None))
    else:
        sys.exit(open_dashboard(sys.argv[1:] or None))
