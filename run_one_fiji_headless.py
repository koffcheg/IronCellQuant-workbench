from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT = Path(r"C:\PERSONAL\ImageJ\IronCells_MVP")
DEFAULT_OUTPUT_ROOT = PROJECT / "output"
DEFAULT_MACRO = PROJECT / "macros" / "Main_IronCells_headless.ijm"
DEFAULT_FIJI = Path(r"C:\PERSONAL\ImageJ\Fiji\fiji.bat")

EXPECTED_OUTPUTS = [
    "final_analysis_overlay.tif",
    "all_components_before_filter.csv",
    "per_object_features.csv",
    "per_image_summary.csv",
    "run_parameters.txt",
    "run_parameters.csv",
    "macro_log.txt",
]

DEFAULT_PARAMS = {
    # Параметри нижче напряму передаються у Fiji macro.
    # Їх можна змінювати з CLI, не редагуючи .ijm файл між експериментами.
    "threshold_method": "Otsu",
    "threshold_mode": "dark",
    "background_rolling": "80",
    "median_radius": "2",
    "contrast_saturated": "0.35",
    "morph_open_iterations": "0",
    "morph_close_iterations": "1",
    "fill_holes": "false",
    "metadata_bar_height": "180",
    "particle_extract_min_area": "10",
    "particle_extract_max_area": "2000000",
    "min_noise_area": "10",
    "min_single_cell_area": "80",
    "max_single_cell_area": "50000",
    "min_aggregate_area": "50000",
    "max_aggregate_area": "2000000",
    "max_single_cell_aspect": "10",
    "max_aggregate_aspect": "30",
    "exclude_border_objects": "true",
    "border_margin_px": "2",
    "blue_min": "120",
    "blue_over_red": "20",
    "blue_over_green": "10",
    "save_overlays": "true",
    "label_objects": "true",
    "contour_width": "6",
    "make_segmentation_sweep": "false",
}


def windows_short_path(path: Path) -> str:
    """Повертає 8.3 short path для Fiji, якщо Unicode-шлях може зламатися у fiji.bat."""
    if os.name != "nt":
        return str(path)
    escaped = str(path).replace("'", "''")
    ps = (
        "$fso = New-Object -ComObject Scripting.FileSystemObject; "
        f"$f = $fso.GetFile('{escaped}'); "
        "$f.ShortPath"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        text=True,
        capture_output=True,
    )
    value = completed.stdout.strip()
    if completed.returncode == 0 and value and Path(value).exists() and '"' not in value:
        return value
    return str(path)


def group_name_for(image: Path) -> str:
    """Назва групи в CSV береться з батьківської папки зображення."""
    if image.parent.name.lower() == "input":
        return "mvp_input"
    return image.parent.name or "unknown"


def discover_default_input() -> Path:
    """Знаходимо тестовий кадр без жорстко прошитого Unicode-рядка в коді."""
    candidates = sorted((PROJECT / "input").glob("52*.bmp"))
    if not candidates:
        return PROJECT / "input" / "52_proto1.bmp"
    for candidate in candidates:
        if "_2" not in candidate.stem:
            return candidate
    return candidates[0]


def build_macro_arg(image: Path, output: Path, params: dict[str, str]) -> tuple[str, str]:
    """Формуємо аргументи macro як key=value; Fiji виконує сам аналіз, Python лише передає параметри."""
    short_input = windows_short_path(image)
    short_used = "true" if short_input != str(image) else "false"
    # original_* поля потрібні для CSV, бо short path стабільний для Fiji,
    # але погано читається людиною і не містить нормальну назву групи/файлу.
    all_params = {
        "input": short_input,
        "output": str(output),
        "original_long_path": str(image),
        "original_file_name": image.name,
        "group_name": group_name_for(image),
        "short_path_used": short_used,
        **params,
    }
    return ";".join(f"{k}={v}" for k, v in all_params.items()), short_used


def create_clean_output(root: Path, prefix: str, clean: bool = False) -> Path:
    """Кожен запуск отримує чисту папку, щоб старі файли не могли видати себе за новий результат."""
    if clean:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        return root
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = root / f"{prefix}_{run_id}"
    counter = 2
    while output.exists():
        output = root / f"{prefix}_{run_id}_{counter}"
        counter += 1
    output.mkdir(parents=True)
    return output


def run_fiji(image: Path, output: Path, params: dict[str, str], fiji: Path, macro: Path, timeout_seconds: int = 180) -> dict[str, str]:
    """Запускає Fiji headless і перевіряє не тільки exit code, а й файли поточного запуску."""
    started = datetime.now()
    macro_arg, _ = build_macro_arg(image, output, params)
    cmd = [str(fiji), "--headless", "-macro", str(macro), macro_arg]
    # Параметри пишемо до старту Fiji: навіть якщо macro зависне, буде видно,
    # з якими налаштуваннями створено цю output-папку.
    write_run_parameters(output, image, params, fiji, macro, macro_arg)
    try:
        # Використовуємо Popen замість subprocess.run, щоб при timeout на Windows
        # прибрати все дерево процесів fiji.bat -> Fiji/ImageJ.
        process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        completed = subprocess.CompletedProcess(cmd, returncode=process.returncode, stdout=stdout, stderr=stderr)
        timed_out = False
    except subprocess.TimeoutExpired:
        # Якщо Fiji зависає, вбиваємо все дерево fiji.bat -> Fiji/ImageJ, а не тільки wrapper.
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, text=True)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        completed = subprocess.CompletedProcess(cmd, returncode=124, stdout=stdout or "", stderr=stderr or "")
        timed_out = True
    finished = datetime.now()
    # stdout/stderr пишуться завжди, навіть якщо Fiji мовчить або повертає 0 без outputs.
    (output / "fiji_stdout.txt").write_text(completed.stdout, encoding="utf-8", errors="replace")
    (output / "fiji_stderr.txt").write_text(completed.stderr, encoding="utf-8", errors="replace")
    (output / "runner_command.txt").write_text(" ".join(cmd), encoding="utf-8", errors="replace")
    append_runner_parameters_to_macro_log(output, params)
    # Перевірка timestamp захищає від stale outputs у папці, яку могли повторно використати вручну.
    # Debug masks та окремі overlay не є обов'язковими, якщо їх вимкнули для економії I/O.
    missing = []
    stale = []
    for name in EXPECTED_OUTPUTS:
        path = output / name
        if not path.exists():
            missing.append(name)
        elif datetime.fromtimestamp(path.stat().st_mtime) < started:
            stale.append(name)
    # Важливо: returncode 0 від Fiji сам по собі не означає успіх.
    # Успіх — це returncode 0 + усі очікувані файли + файли новіші за час старту.
    ok = completed.returncode == 0 and not missing and not stale
    return {
        "image": str(image),
        "output": str(output),
        "returncode": str(completed.returncode),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "missing_outputs": ";".join(missing),
        "stale_outputs": ";".join(stale),
        "timed_out": str(timed_out),
        "ok": str(ok),
    }


def write_run_parameters(output: Path, image: Path, params: dict[str, str], fiji: Path, macro: Path, macro_arg: str) -> None:
    """Пишемо параметри runner'ом, щоб macro не зупинявся на довгих File.append рядках."""
    lines = [
        f"input={image}",
        f"output={output}",
        f"macro={macro}",
        f"fiji={fiji}",
        f"macro_argument_string={macro_arg}",
    ]
    lines.extend(f"{key}={value}" for key, value in params.items())
    (output / "run_parameters.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "run_parameters.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        writer.writerow(["input", str(image)])
        writer.writerow(["output", str(output)])
        writer.writerow(["macro", str(macro)])
        writer.writerow(["fiji", str(fiji)])
        for key, value in params.items():
            writer.writerow([key, value])


def append_runner_parameters_to_macro_log(output: Path, params: dict[str, str]) -> None:
    """Додаємо параметри в macro_log після Fiji, щоб macro не ламався на довгих log-рядках."""
    log_path = output / "macro_log.txt"
    with log_path.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\nRunner parameter summary:\n")
        for key, value in params.items():
            f.write(f"{key}={value}\n")


def add_param_args(parser: argparse.ArgumentParser) -> None:
    """Додаємо всі ключові параметри Fiji macro без редагування коду між експериментами."""
    for key, value in DEFAULT_PARAMS.items():
        parser.add_argument("--" + key.replace("_", "-"), default=value)


def params_from_args(args: argparse.Namespace) -> dict[str, str]:
    """Збирає тільки macro-параметри, не змішуючи їх зі службовими CLI полями runner'а."""
    return {key: str(getattr(args, key)) for key in DEFAULT_PARAMS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Fiji/ImageJ IronCells analysis into a fresh output folder.")
    parser.add_argument("--input", type=Path, default=discover_default_input())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, help="Exact output folder. Use with --clean-output to intentionally reuse it.")
    parser.add_argument("--clean-output", action="store_true", help="Delete and recreate --output before running.")
    parser.add_argument("--prefix", default="single_test")
    parser.add_argument("--macro", type=Path, default=DEFAULT_MACRO)
    parser.add_argument("--fiji", type=Path, default=DEFAULT_FIJI)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    add_param_args(parser)
    return parser.parse_args()


def main() -> int:
    # main() не аналізує зображення. Він тільки валідує шляхи, створює output і запускає Fiji.
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if not args.fiji.exists():
        raise SystemExit(f"Fiji runner not found: {args.fiji}")
    if not args.macro.exists():
        raise SystemExit(f"Macro not found: {args.macro}")

    output = create_clean_output(args.output or args.output_root, args.prefix, clean=args.clean_output and args.output is not None)
    row = run_fiji(args.input, output, params_from_args(args), args.fiji, args.macro, timeout_seconds=args.timeout_seconds)
    with (output / "runner_report.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"ok={row['ok']} returncode={row['returncode']} output={output}")
    if row["missing_outputs"]:
        print(f"missing={row['missing_outputs']}")
    if row["stale_outputs"]:
        print(f"stale={row['stale_outputs']}")
    return 0 if row["ok"] == "True" else 1


if __name__ == "__main__":
    raise SystemExit(main())
