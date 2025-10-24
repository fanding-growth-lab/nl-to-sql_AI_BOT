#!/usr/bin/env python3
"""
문서 자동 생성 스크립트

이 스크립트는 프로젝트의 소스 코드를 분석하여 자동으로 API 문서를 생성합니다.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_path():
    """프로젝트 루트를 Python 경로에 추가"""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

def install_docs_requirements():
    """문서 생성에 필요한 패키지 설치"""
    docs_requirements = Path(__file__).parent.parent / "docs" / "requirements.txt"
    
    if docs_requirements.exists():
        print("문서 생성 패키지 설치 중...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(docs_requirements)
        ], check=True)
    else:
        print("문서 요구사항 파일을 찾을 수 없습니다.")

def generate_api_docs():
    """API 문서 자동 생성"""
    print("API 문서 생성 중...")
    
    # sphinx-apidoc를 사용하여 API 문서 생성
    docs_dir = Path(__file__).parent.parent / "docs"
    source_dir = docs_dir / "api"
    
    # API 문서 디렉토리 생성
    source_dir.mkdir(exist_ok=True)
    
    # src 디렉토리에서 API 문서 생성
    src_dir = Path(__file__).parent.parent / "src"
    
    subprocess.run([
        "sphinx-apidoc",
        "-f",  # 강제 생성
        "-o", str(source_dir),
        str(src_dir),
        "--separate",  # 각 모듈별로 파일 생성
        "--module-first",  # 모듈을 먼저 설명
    ], check=True)
    
    print(f"API 문서가 {source_dir}에 생성되었습니다.")

def build_docs():
    """문서 빌드"""
    print("문서 빌드 중...")
    
    docs_dir = Path(__file__).parent.parent / "docs"
    
    # HTML 문서 빌드
    subprocess.run([
        "sphinx-build",
        "-b", "html",
        str(docs_dir),
        str(docs_dir / "_build" / "html"),
        "-W",  # 경고를 오류로 처리
    ], check=True)
    
    print("HTML 문서가 docs/_build/html에 생성되었습니다.")

def build_pdf():
    """PDF 문서 빌드"""
    print("PDF 문서 빌드 중...")
    
    docs_dir = Path(__file__).parent.parent / "docs"
    
    try:
        subprocess.run([
            "sphinx-build",
            "-b", "latex",
            str(docs_dir),
            str(docs_dir / "_build" / "latex"),
        ], check=True)
        
        # LaTeX를 PDF로 변환
        latex_dir = docs_dir / "_build" / "latex"
        subprocess.run([
            "make", "-C", str(latex_dir), "pdf"
        ], check=True)
        
        print("PDF 문서가 docs/_build/latex에 생성되었습니다.")
    except subprocess.CalledProcessError:
        print("PDF 빌드에 실패했습니다. LaTeX가 설치되어 있는지 확인하세요.")

def clean_docs():
    """문서 빌드 파일 정리"""
    print("문서 빌드 파일 정리 중...")
    
    docs_dir = Path(__file__).parent.parent / "docs"
    build_dir = docs_dir / "_build"
    
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
        print("빌드 파일이 정리되었습니다.")

def validate_docs():
    """문서 유효성 검사"""
    print("문서 유효성 검사 중...")
    
    docs_dir = Path(__file__).parent.parent / "docs"
    
    # Sphinx 링크 검사
    subprocess.run([
        "sphinx-build",
        "-b", "linkcheck",
        str(docs_dir),
        str(docs_dir / "_build" / "linkcheck"),
    ], check=True)
    
    print("링크 검사가 완료되었습니다.")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="DataTalk 문서 자동 생성")
    parser.add_argument("--clean", action="store_true", help="빌드 파일 정리")
    parser.add_argument("--api", action="store_true", help="API 문서만 생성")
    parser.add_argument("--html", action="store_true", help="HTML 문서 빌드")
    parser.add_argument("--pdf", action="store_true", help="PDF 문서 빌드")
    parser.add_argument("--validate", action="store_true", help="문서 유효성 검사")
    parser.add_argument("--all", action="store_true", help="모든 작업 수행")
    
    args = parser.parse_args()
    
    if args.clean:
        clean_docs()
        return
    
    if args.all or not any([args.api, args.html, args.pdf, args.validate]):
        # 기본적으로 모든 작업 수행
        install_docs_requirements()
        generate_api_docs()
        build_docs()
        validate_docs()
    else:
        if args.api:
            install_docs_requirements()
            generate_api_docs()
        
        if args.html:
            install_docs_requirements()
            build_docs()
        
        if args.pdf:
            install_docs_requirements()
            build_pdf()
        
        if args.validate:
            validate_docs()

if __name__ == "__main__":
    main()

