DataTalk Documentation
=====================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   installation
   user_guide
   api_reference
   development
   testing
   deployment
   troubleshooting

Welcome to DataTalk's documentation!
====================================

DataTalk은 자연어를 SQL로 변환하는 지능형 챗봇 시스템입니다. 
사용자가 자연어로 질문하면 자동으로 적절한 SQL 쿼리를 생성하고 실행하여 결과를 제공합니다.

주요 특징
---------

* **자연어 이해**: 한국어 자연어 쿼리를 정확히 해석
* **지능형 SQL 생성**: LLM과 RAG 기반의 정확한 SQL 생성
* **실시간 응답**: Slack을 통한 즉시 응답
* **확장 가능**: 모듈화된 아키텍처로 쉬운 확장
* **고성능**: 최적화된 쿼리 실행과 캐싱

빠른 시작
---------

1. 저장소 클론::

   git clone https://github.com/your-org/datatalk.git
   cd datatalk

2. 가상 환경 설정::

   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

3. 의존성 설치::

   pip install -r requirements.txt

4. 환경 변수 설정::

   cp env.template .env
   # .env 파일을 편집하여 API 키 설정

5. 애플리케이션 실행::

   python src/main.py

더 자세한 내용은 :doc:`installation` 가이드를 참조하세요.

시스템 아키텍처
--------------

.. mermaid::
   :align: center

   graph TB
       A[Slack 사용자] --> B[Slack Bot API]
       B --> C[FastAPI 서버]
       C --> D[NL-to-SQL 파이프라인]
       
       D --> E[인텐트 분류기]
       D --> F[엔티티 추출기]
       D --> G[스키마 매퍼]
       D --> H[SQL 생성기]
       D --> I[SQL 검증기]
       
       H --> J[동적 SQL 생성]
       H --> K[RAG 템플릿]
       
       I --> L[SQL 실행기]
       L --> M[데이터베이스]
       
       M --> N[결과 처리]
       N --> O[응답 생성]
       O --> B

핵심 컴포넌트
-------------

* **NLProcessor**: 자연어 처리 및 인텐트 분류
* **SchemaMapper**: 자연어를 데이터베이스 스키마에 매핑
* **DynamicSQLGenerator**: LLM 기반 SQL 생성
* **SQLValidationNode**: SQL 검증 및 최적화
* **AsyncSQLExecutor**: 비동기 SQL 실행

사용 예시
---------

.. code-block:: text

   사용자: "8월 신규 가입자 수를 알려줘"
   봇: "8월 신규 가입자 수: 1,234명"

   사용자: "상위 5 크리에이터의 회원 수를 알려줘"
   봇: "상위 5 크리에이터 회원 수:
        1. 크리에이터A: 5,678명
        2. 크리에이터B: 4,567명
        ..."

문서 구조
---------

* :doc:`overview` - 시스템 개요 및 아키텍처
* :doc:`installation` - 설치 및 설정 가이드
* :doc:`user_guide` - 사용자 가이드
* :doc:`api_reference` - API 참조 문서
* :doc:`development` - 개발자 가이드
* :doc:`testing` - 테스트 가이드
* :doc:`deployment` - 배포 가이드
* :doc:`troubleshooting` - 문제 해결 가이드

기여하기
--------

DataTalk 프로젝트에 기여하고 싶으시다면:

1. `GitHub 저장소 <https://github.com/your-org/datatalk>`_ 를 Fork하세요
2. 기능 브랜치를 생성하세요 (``git checkout -b feature/amazing-feature``)
3. 변경사항을 커밋하세요 (``git commit -m 'Add amazing feature'``)
4. 브랜치를 푸시하세요 (``git push origin feature/amazing-feature``)
5. Pull Request를 생성하세요

라이선스
--------

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 
자세한 내용은 `LICENSE <https://github.com/your-org/datatalk/blob/main/LICENSE>`_ 파일을 참조하세요.

지원
----

* **이슈 트래커**: `GitHub Issues <https://github.com/your-org/datatalk/issues>`_
* **문서**: `프로젝트 문서 <https://datatalk.readthedocs.io/>`_
* **이메일**: support@datatalk.com

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

