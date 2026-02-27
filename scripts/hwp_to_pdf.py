"""
HWP 파일을 PDF로 변환하는 스크립트.

필요 조건: LibreOffice 설치
- Windows: https://www.libreoffice.org/ 에서 설치
- Linux (Ubuntu/Debian): sudo apt install libreoffice
- Linux (RHEL/CentOS): sudo yum install libreoffice
"""

import os
import subprocess
import sys
from pathlib import Path


def _clean_env_for_libreoffice() -> dict[str, str] | None:
    """LibreOffice 실행 시 Python(Anaconda) 환경 충돌 방지: PYTHON* 제거."""
    if sys.platform != "win32":
        return None
    env = os.environ.copy()
    for key in list(env):
        if key.upper().startswith("PYTHON"):
            env.pop(key, None)
    return env


def _get_short_path(path: Path) -> Path:
    """Windows에서 한글/공백 경로 문제 방지: 8.3 짧은 경로 반환."""
    if sys.platform != "win32":
        return path
    try:
        import ctypes
        p = path.resolve()
        if not p.exists():
            return path
        buf = ctypes.create_unicode_buffer(1024)
        r = ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, 1024)
        if r and buf.value:
            return Path(buf.value)
    except Exception:
        pass
    return path


def get_libreoffice_cmd() -> list[str]:
    """OS에 맞는 LibreOffice 실행 명령 반환."""
    if sys.platform == "win32":
        candidates = [
            Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
            Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
        ]
        for p in candidates:
            if p.exists():
                return [str(p)]
        return ["soffice"]
    return ["libreoffice", "--headless"]


def convert_hwp_to_pdf(
    input_dir: Path,
    output_dir: Path | None = None,
    libreoffice_cmd: list[str] | None = None,
) -> list[Path]:
    """
    지정 디렉토리 내 모든 .hwp 파일을 PDF로 변환.

    Args:
        input_dir: HWP 파일이 있는 디렉토리
        output_dir: PDF 저장 디렉토리 (None이면 input_dir과 동일)
        libreoffice_cmd: LibreOffice 실행 명령 (None이면 자동 탐지)

    Returns:
        변환된 PDF 파일 경로 목록
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    hwp_files = sorted(input_dir.glob("*.hwp"))
    if not hwp_files:
        print(f"[INFO] .hwp 파일이 없습니다: {input_dir}")
        return []

    cmd = libreoffice_cmd or get_libreoffice_cmd()
    base = cmd if "--headless" in cmd else cmd + ["--headless"]

    run_env = _clean_env_for_libreoffice()
    outdir_str = str(_get_short_path(output_dir))

    result_paths: list[Path] = []
    for hwp_path in hwp_files:
        try:
            # Windows: 한글 경로 인코딩 문제 방지를 위해 짧은 경로 사용
            hwp_arg = str(_get_short_path(hwp_path))
            args = base + [
                "--convert-to", "pdf",
                "--outdir", outdir_str,
                hwp_arg,
            ]
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=120,
                env=run_env,
            )
            if proc.returncode != 0:
                print(f"[WARN] 변환 실패: {hwp_path.name} - {proc.stderr or proc.stdout}")
                continue
            pdf_path = output_dir / (hwp_path.stem + ".pdf")
            if pdf_path.exists():
                result_paths.append(pdf_path)
                print(f"[OK] {hwp_path.name} -> {pdf_path.name}")
        except subprocess.TimeoutExpired:
            print(f"[WARN] 타임아웃: {hwp_path.name}")
        except Exception as e:
            print(f"[WARN] 오류 ({hwp_path.name}): {e}")

    return result_paths


def main():
    # 프로젝트 루트 기준 data/raw/.../files 경로
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_input = (
        project_root
        / "data"
        / "raw"
        / "원본 데이터-20260206T023549Z-1-001"
        / "원본 데이터"
        / "files"
    )

    if len(sys.argv) >= 2:
        input_dir = Path(sys.argv[1]).resolve()
    else:
        input_dir = default_input

    output_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else None

    if not input_dir.exists():
        print(f"[ERROR] 디렉토리가 없습니다: {input_dir}")
        sys.exit(1)

    print(f"입력 디렉토리: {input_dir}")
    paths = convert_hwp_to_pdf(input_dir, output_dir)
    print(f"변환 완료: {len(paths)}개 PDF")
    sys.exit(0 if paths else 1)


if __name__ == "__main__":
    main()
