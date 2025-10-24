#!/usr/bin/env python3
"""
문서 품질 검사 스크립트

이 스크립트는 생성된 문서의 품질을 검사합니다.
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any

def setup_path():
    """프로젝트 루트를 Python 경로에 추가"""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

def check_markdown_syntax(file_path: Path) -> List[str]:
    """Markdown 파일의 문법 오류 검사"""
    errors = []
    
    if not file_path.exists():
        errors.append(f"파일이 존재하지 않습니다: {file_path}")
        return errors
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        # 제목 레벨 검사
        if line.startswith('#'):
            if line.startswith('#######'):
                errors.append(f"{file_path}:{i} - 제목 레벨이 너무 깊습니다 (7개 이상의 #)")
        
        # 링크 문법 검사
        if '](' in line and not line.strip().startswith('#'):
            if not re.search(r'\[.*?\]\(.*?\)', line):
                errors.append(f"{file_path}:{i} - 잘못된 링크 문법: {line.strip()}")
        
        # 이미지 문법 검사
        if '![' in line:
            if not re.search(r'!\[.*?\]\(.*?\)', line):
                errors.append(f"{file_path}:{i} - 잘못된 이미지 문법: {line.strip()}")
        
        # 코드 블록 검사
        if line.strip().startswith('```'):
            if not line.strip().endswith('```') and len(line.strip()) > 3:
                # 코드 블록 시작
                pass
            elif line.strip() == '```':
                # 코드 블록 종료
                pass
    
    return errors

def check_rst_syntax(file_path: Path) -> List[str]:
    """reStructuredText 파일의 문법 오류 검사"""
    errors = []
    
    if not file_path.exists():
        errors.append(f"파일이 존재하지 않습니다: {file_path}")
        return errors
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        # 제목 검사
        if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            if i < len(lines) - 1:
                next_line = lines[i]
                if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t'):
                    # 제목인지 확인
                    if line.strip() and not line.startswith('..') and not line.startswith('|'):
                        # 다음 줄이 제목 구분선인지 확인
                        if not (next_line.startswith('=') or next_line.startswith('-') or next_line.startswith('~')):
                            # 제목이 아닌 경우
                            pass
        
        # 링크 검사
        if '`' in line and '`_' in line:
            if not re.search(r'`.*?`_', line):
                errors.append(f"{file_path}:{i} - 잘못된 링크 문법: {line.strip()}")
        
        # 지시문 검사
        if line.strip().startswith('.. '):
            directive = line.strip()[3:].split('::')[0].split()[0]
            valid_directives = [
                'toctree', 'code-block', 'literalinclude', 'include',
                'image', 'figure', 'table', 'list-table', 'csv-table',
                'note', 'warning', 'tip', 'important', 'danger',
                'admonition', 'attention', 'caution', 'error', 'hint',
                'versionadded', 'versionchanged', 'deprecated',
                'seealso', 'todo', 'glossary', 'rubric', 'sidebar',
                'topic', 'parsed-literal', 'epigraph', 'highlights',
                'pull-quote', 'compound', 'container', 'raw', 'replace',
                'unicode', 'date', 'include', 'literalinclude', 'code-block',
                'math', 'role', 'default-role', 'title', 'subtitle',
                'section', 'subsection', 'subsubsection', 'paragraph',
                'contents', 'sectnum', 'header', 'footer', 'footnotes',
                'citations', 'target-notes', 'line-block', 'doctest',
                'testcode', 'testoutput', 'testsetup', 'testcleanup',
                'test', 'testsetup', 'testcleanup', 'testcode', 'testoutput',
                'test', 'testsetup', 'testcleanup', 'testcode', 'testoutput'
            ]
            if directive not in valid_directives:
                errors.append(f"{file_path}:{i} - 알 수 없는 지시문: {directive}")
    
    return errors

def check_documentation_structure(docs_dir: Path) -> List[str]:
    """문서 구조 검사"""
    errors = []
    
    required_files = [
        "README.md",
        "docs/index.rst",
        "docs/overview.md",
        "docs/installation.md",
        "docs/user_guide.md",
        "docs/api_reference.md",
        "docs/development.md",
        "docs/testing.md",
        "docs/deployment.md",
        "docs/troubleshooting.md"
    ]
    
    for file_path in required_files:
        full_path = docs_dir / file_path
        if not full_path.exists():
            errors.append(f"필수 문서 파일이 없습니다: {file_path}")
    
    return errors

def check_code_examples(docs_dir: Path) -> List[str]:
    """코드 예제 검사"""
    errors = []
    
    # Python 코드 블록 검사
    for file_path in docs_dir.rglob("*.md"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 코드 블록 찾기
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        
        for i, code_block in enumerate(code_blocks):
            lines = code_block.split('\n')
            for j, line in enumerate(lines):
                # 들여쓰기 검사
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    if j > 0 and lines[j-1].strip():
                        # 이전 줄이 있고 현재 줄이 들여쓰기 없이 시작하는 경우
                        if not line.startswith('def ') and not line.startswith('class ') and not line.startswith('import ') and not line.startswith('from '):
                            errors.append(f"{file_path} - 코드 블록 {i+1}, 줄 {j+1}: 들여쓰기 오류")
    
    return errors

def check_links(docs_dir: Path) -> List[str]:
    """링크 검사"""
    errors = []
    
    for file_path in docs_dir.rglob("*.md"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 링크 찾기
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        
        for link_text, link_url in links:
            if link_url.startswith('http'):
                # 외부 링크는 검사하지 않음
                continue
            elif link_url.startswith('#'):
                # 앵커 링크는 검사하지 않음
                continue
            elif link_url.startswith('mailto:'):
                # 이메일 링크는 검사하지 않음
                continue
            else:
                # 상대 경로 링크 검사
                if link_url.startswith('./'):
                    link_url = link_url[2:]
                elif link_url.startswith('../'):
                    # 상위 디렉토리 링크
                    continue
                
                # 링크 대상 파일 검사
                target_path = file_path.parent / link_url
                if not target_path.exists():
                    errors.append(f"{file_path} - 링크 대상이 존재하지 않습니다: {link_url}")
    
    return errors

def check_spelling(docs_dir: Path) -> List[str]:
    """맞춤법 검사 (기본적인 한국어 오타 검사)"""
    errors = []
    
    # 일반적인 오타 패턴
    typo_patterns = [
        (r'데이타', '데이터'),
        (r'프로그래밍', '프로그래밍'),
        (r'설정', '설정'),
        (r'사용자', '사용자'),
        (r'시스템', '시스템'),
        (r'데이터베이스', '데이터베이스'),
        (r'애플리케이션', '애플리케이션'),
        (r'인터페이스', '인터페이스'),
        (r'프로세스', '프로세스'),
        (r'컴포넌트', '컴포넌트'),
        (r'아키텍처', '아키텍처'),
        (r'알고리즘', '알고리즘'),
        (r'메모리', '메모리'),
        (r'프로세서', '프로세서'),
        (r'네트워크', '네트워크'),
        (r'프로토콜', '프로토콜'),
        (r'알고리즘', '알고리즘'),
        (r'데이터베이스', '데이터베이스'),
        (r'애플리케이션', '애플리케이션'),
        (r'인터페이스', '인터페이스'),
        (r'프로세스', '프로세스'),
        (r'컴포넌트', '컴포넌트'),
        (r'아키텍처', '아키텍처'),
        (r'알고리즘', '알고리즘'),
        (r'메모리', '메모리'),
        (r'프로세서', '프로세서'),
        (r'네트워크', '네트워크'),
        (r'프로토콜', '프로토콜')
    ]
    
    for file_path in docs_dir.rglob("*.md"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for pattern, correct in typo_patterns:
            if re.search(pattern, content):
                errors.append(f"{file_path} - 오타 발견: '{pattern}' -> '{correct}'")
    
    return errors

def generate_report(errors: List[str], output_file: Path = None):
    """검사 결과 보고서 생성"""
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 문서 품질 검사 보고서\n\n")
            f.write(f"검사 일시: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"총 오류 수: {len(errors)}\n\n")
            
            if errors:
                f.write("## 발견된 오류\n\n")
                for i, error in enumerate(errors, 1):
                    f.write(f"{i}. {error}\n")
            else:
                f.write("## 검사 결과\n\n")
                f.write("오류가 발견되지 않았습니다. 문서 품질이 양호합니다.\n")
    else:
        print("# 문서 품질 검사 보고서")
        print(f"검사 일시: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 오류 수: {len(errors)}")
        
        if errors:
            print("\n## 발견된 오류")
            for i, error in enumerate(errors, 1):
                print(f"{i}. {error}")
        else:
            print("\n## 검사 결과")
            print("오류가 발견되지 않았습니다. 문서 품질이 양호합니다.")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="DataTalk 문서 품질 검사")
    parser.add_argument("--output", "-o", help="보고서 출력 파일")
    parser.add_argument("--docs-dir", default="docs", help="문서 디렉토리 경로")
    
    args = parser.parse_args()
    
    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"문서 디렉토리를 찾을 수 없습니다: {docs_dir}")
        sys.exit(1)
    
    print("문서 품질 검사 시작...")
    
    all_errors = []
    
    # Markdown 파일 검사
    print("Markdown 파일 검사 중...")
    for file_path in docs_dir.rglob("*.md"):
        errors = check_markdown_syntax(file_path)
        all_errors.extend(errors)
    
    # reStructuredText 파일 검사
    print("reStructuredText 파일 검사 중...")
    for file_path in docs_dir.rglob("*.rst"):
        errors = check_rst_syntax(file_path)
        all_errors.extend(errors)
    
    # 문서 구조 검사
    print("문서 구조 검사 중...")
    errors = check_documentation_structure(docs_dir)
    all_errors.extend(errors)
    
    # 코드 예제 검사
    print("코드 예제 검사 중...")
    errors = check_code_examples(docs_dir)
    all_errors.extend(errors)
    
    # 링크 검사
    print("링크 검사 중...")
    errors = check_links(docs_dir)
    all_errors.extend(errors)
    
    # 맞춤법 검사
    print("맞춤법 검사 중...")
    errors = check_spelling(docs_dir)
    all_errors.extend(errors)
    
    # 보고서 생성
    output_file = Path(args.output) if args.output else None
    generate_report(all_errors, output_file)
    
    if all_errors:
        print(f"\n총 {len(all_errors)}개의 오류가 발견되었습니다.")
        sys.exit(1)
    else:
        print("\n문서 품질 검사가 완료되었습니다. 오류가 발견되지 않았습니다.")
        sys.exit(0)

if __name__ == "__main__":
    main()

