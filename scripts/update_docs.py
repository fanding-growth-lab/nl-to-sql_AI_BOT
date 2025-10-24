#!/usr/bin/env python3
"""
문서 자동 업데이트 스크립트

이 스크립트는 프로젝트의 변경사항을 감지하여 관련 문서를 자동으로 업데이트합니다.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def setup_path():
    """프로젝트 루트를 Python 경로에 추가"""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

def get_git_changes():
    """Git 변경사항 감지"""
    try:
        # 최근 커밋의 변경된 파일 목록
        result = subprocess.run([
            "git", "diff", "--name-only", "HEAD~1", "HEAD"
        ], capture_output=True, text=True, check=True)
        
        changed_files = result.stdout.strip().split('\n')
        return [f for f in changed_files if f and not f.startswith('docs/')]
    except subprocess.CalledProcessError:
        print("Git 변경사항을 감지할 수 없습니다.")
        return []

def analyze_code_changes(changed_files: List[str]) -> Dict[str, Any]:
    """코드 변경사항 분석"""
    changes = {
        'new_files': [],
        'modified_files': [],
        'deleted_files': [],
        'api_changes': [],
        'config_changes': [],
        'test_changes': []
    }
    
    for file_path in changed_files:
        if not file_path:
            continue
            
        if file_path.startswith('src/'):
            if 'test' in file_path.lower():
                changes['test_changes'].append(file_path)
            elif file_path.endswith('.py'):
                changes['modified_files'].append(file_path)
        elif file_path.startswith('config/') or file_path.endswith('.json') or file_path.endswith('.yaml'):
            changes['config_changes'].append(file_path)
        elif file_path.endswith('.py') and 'api' in file_path.lower():
            changes['api_changes'].append(file_path)
    
    return changes

def update_api_docs(api_changes: List[str]):
    """API 문서 업데이트"""
    if not api_changes:
        return
    
    print("API 문서 업데이트 중...")
    
    # sphinx-apidoc를 사용하여 API 문서 재생성
    docs_dir = Path(__file__).parent.parent / "docs"
    source_dir = docs_dir / "api"
    src_dir = Path(__file__).parent.parent / "src"
    
    subprocess.run([
        "sphinx-apidoc",
        "-f",  # 강제 생성
        "-o", str(source_dir),
        str(src_dir),
        "--separate",
        "--module-first",
    ], check=True)
    
    print(f"API 문서가 업데이트되었습니다: {len(api_changes)}개 파일")

def update_installation_guide(config_changes: List[str]):
    """설치 가이드 업데이트"""
    if not config_changes:
        return
    
    print("설치 가이드 업데이트 중...")
    
    # 설정 파일 변경사항을 설치 가이드에 반영
    docs_dir = Path(__file__).parent.parent / "docs"
    installation_file = docs_dir / "installation.md"
    
    if installation_file.exists():
        with open(installation_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 설정 파일 변경사항 추가
        if "설정 파일 변경사항" not in content:
            content += "\n\n## 설정 파일 변경사항\n\n"
            content += f"최근 업데이트 ({datetime.now().strftime('%Y-%m-%d')}):\n\n"
            for config_file in config_changes:
                content += f"- {config_file}\n"
        
        with open(installation_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("설치 가이드가 업데이트되었습니다.")

def update_development_guide(test_changes: List[str]):
    """개발 가이드 업데이트"""
    if not test_changes:
        return
    
    print("개발 가이드 업데이트 중...")
    
    # 테스트 파일 변경사항을 개발 가이드에 반영
    docs_dir = Path(__file__).parent.parent / "docs"
    development_file = docs_dir / "development.md"
    
    if development_file.exists():
        with open(development_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 테스트 변경사항 추가
        if "테스트 변경사항" not in content:
            content += "\n\n## 테스트 변경사항\n\n"
            content += f"최근 업데이트 ({datetime.now().strftime('%Y-%m-%d')}):\n\n"
            for test_file in test_changes:
                content += f"- {test_file}\n"
        
        with open(development_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("개발 가이드가 업데이트되었습니다.")

def update_changelog(changes: Dict[str, Any]):
    """CHANGELOG 업데이트"""
    print("CHANGELOG 업데이트 중...")
    
    changelog_file = Path(__file__).parent.parent / "CHANGELOG.md"
    
    # CHANGELOG 파일이 없으면 생성
    if not changelog_file.exists():
        with open(changelog_file, 'w', encoding='utf-8') as f:
            f.write("# Changelog\n\n")
            f.write("All notable changes to this project will be documented in this file.\n\n")
            f.write("The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),\n")
            f.write("and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).\n\n")
    
    # 최신 변경사항 추가
    with open(changelog_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 새로운 섹션 추가
    today = datetime.now().strftime('%Y-%m-%d')
    
    if f"## [{today}]" not in content:
        new_section = f"\n## [{today}]\n\n"
        
        if changes['api_changes']:
            new_section += "### Added\n"
            for api_file in changes['api_changes']:
                new_section += f"- Updated API: {api_file}\n"
            new_section += "\n"
        
        if changes['config_changes']:
            new_section += "### Changed\n"
            for config_file in changes['config_changes']:
                new_section += f"- Updated configuration: {config_file}\n"
            new_section += "\n"
        
        if changes['test_changes']:
            new_section += "### Testing\n"
            for test_file in changes['test_changes']:
                new_section += f"- Updated tests: {test_file}\n"
            new_section += "\n"
        
        # 기존 내용 앞에 새 섹션 추가
        lines = content.split('\n')
        if lines[0] == "# Changelog":
            lines.insert(2, new_section.strip())
        else:
            lines.insert(0, new_section.strip())
        
        content = '\n'.join(lines)
    
    with open(changelog_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("CHANGELOG가 업데이트되었습니다.")

def update_readme():
    """README 업데이트"""
    print("README 업데이트 중...")
    
    readme_file = Path(__file__).parent.parent / "README.md"
    
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 마지막 업데이트 날짜 업데이트
        today = datetime.now().strftime('%Y-%m-%d')
        
        # 마지막 업데이트 날짜 찾기 및 업데이트
        if "마지막 업데이트:" in content:
            content = re.sub(
                r"마지막 업데이트: \d{4}-\d{2}-\d{2}",
                f"마지막 업데이트: {today}",
                content
            )
        else:
            # 마지막 업데이트 날짜 추가
            content += f"\n\n---\n\n마지막 업데이트: {today}\n"
        
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("README가 업데이트되었습니다.")

def validate_updated_docs():
    """업데이트된 문서 검증"""
    print("업데이트된 문서 검증 중...")
    
    # 문서 품질 검사 실행
    check_script = Path(__file__).parent / "check_docs.py"
    
    if check_script.exists():
        try:
            subprocess.run([sys.executable, str(check_script)], check=True)
            print("문서 검증이 완료되었습니다.")
        except subprocess.CalledProcessError:
            print("문서 검증에서 오류가 발견되었습니다.")
    else:
        print("문서 검증 스크립트를 찾을 수 없습니다.")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="DataTalk 문서 자동 업데이트")
    parser.add_argument("--force", action="store_true", help="강제 업데이트")
    parser.add_argument("--validate", action="store_true", help="업데이트 후 검증")
    
    args = parser.parse_args()
    
    print("문서 자동 업데이트 시작...")
    
    # Git 변경사항 감지
    changed_files = get_git_changes()
    
    if not changed_files and not args.force:
        print("변경된 파일이 없습니다. 문서 업데이트를 건너뜁니다.")
        return
    
    # 변경사항 분석
    changes = analyze_code_changes(changed_files)
    
    print(f"변경된 파일 수: {len(changed_files)}")
    print(f"API 변경: {len(changes['api_changes'])}")
    print(f"설정 변경: {len(changes['config_changes'])}")
    print(f"테스트 변경: {len(changes['test_changes'])}")
    
    # 관련 문서 업데이트
    if changes['api_changes']:
        update_api_docs(changes['api_changes'])
    
    if changes['config_changes']:
        update_installation_guide(changes['config_changes'])
    
    if changes['test_changes']:
        update_development_guide(changes['test_changes'])
    
    # CHANGELOG 업데이트
    update_changelog(changes)
    
    # README 업데이트
    update_readme()
    
    # 문서 검증
    if args.validate:
        validate_updated_docs()
    
    print("문서 자동 업데이트가 완료되었습니다.")

if __name__ == "__main__":
    import re
    main()

