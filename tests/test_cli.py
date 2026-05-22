from pipedream import cli


def test_run_example_ir(capsys):
    rc = cli.main(["run", "examples/orders.ir.json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "revenue" in out and "order_count" in out
    assert "2026-02" in out


def test_run_with_csv_output(tmp_path, capsys):
    out_path = tmp_path / "result.csv"
    rc = cli.main(["run", "examples/orders.ir.json", "-o", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    header = out_path.read_text().splitlines()[0]
    assert header == "month,revenue,order_count"


def test_explain_example_ir(capsys):
    rc = cli.main(["explain", "examples/orders.ir.json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "pipeline: monthly_completed_revenue" in out
    assert "load CSV" in out
    assert "output: ranked" in out


def test_run_missing_file_reports_error(capsys):
    rc = cli.main(["run", "examples/does_not_exist.json"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "error:" in err


def test_run_unknown_executor(capsys):
    rc = cli.main(["run", "examples/orders.ir.json", "--executor", "nope"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "unknown executor" in err
