"""
RAG Schema Retriever Node for SQL Generation

This module implements a Retrieval-Augmented Generation (RAG) system
for retrieving relevant database schema information from data_dictionary.md
to improve SQL generation accuracy.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .state import GraphState, Entity, SchemaMapping
from .nodes import BaseNode, TABLE_NAME_MAPPING
from core.config import get_settings
from core.db import get_cached_db_schema
from core.logging import get_logger


logger = get_logger(__name__)


class RAGSchemaRetrieverNode(BaseNode):
    """
    RAG-based Schema Retriever Node
    
    Retrieves relevant database schema information from data_dictionary.md
    using vector similarity search to provide context for SQL generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 데이터 사전 파일 경로 설정
        self.schema_path = config.get(
            "schema_path",
            Path(__file__).parent / "data_dictionary.md"
        )
        
        # 벡터 스토어 및 임베딩 초기화
        self.vector_store: Optional[FAISS] = None
        self.embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
        
        # 텍스트 분할기 설정
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        
        # SchemaMapper 통합: DB 스키마 및 테이블명 매핑
        # db_schema가 없거나 비어있으면 초기화 시점에 한 번만 로드 (성능 최적화)
        self.db_schema = config.get("db_schema") or {}
        if not self.db_schema or len(self.db_schema) == 0:
            self.db_schema = get_cached_db_schema()
            self.logger.debug("db_schema was empty in config, loaded from cache during initialization")
        self.table_name_mapping = TABLE_NAME_MAPPING
        
        # 초기화
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG components (embeddings, vector store)"""
        try:
            settings = get_settings()
            
            # Google Generative AI Embeddings 초기화
            from pydantic import SecretStr
            
            api_key_value = settings.llm.api_key
            if not api_key_value:
                logger.warning("Google API key not found, RAG will be disabled")
                return
            
            # Handle SecretStr or plain string
            if hasattr(api_key_value, 'get_secret_value'):
                api_key_str = api_key_value.get_secret_value()  # type: ignore[attr-defined]
            else:
                api_key_str = str(api_key_value)
            
            # Try both parameter names for compatibility
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=SecretStr(api_key_str)
                )
            except TypeError:
                # Fallback to api_key parameter
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    api_key=SecretStr(api_key_str)
                )
            
            # 벡터 스토어 로드 또는 생성
            self._load_or_create_vector_store()
            
            logger.info("RAGSchemaRetrieverNode initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
            self.vector_store = None
            self.embeddings = None
    
    def _enhance_query_for_embedding(self, query: str, schema_mapping: Optional[Any] = None) -> str:
        """
        Enhance query text for better embedding generation
        Adds SQL/schema-specific context to improve relevance
        
        Args:
            query: Original user query
            schema_mapping: Optional schema mapping with relevant tables/columns
        
        Returns:
            Enhanced query string
        """
        enhanced_parts = [query]
        
        # SQL/스키마 관련 키워드 추가 (임베딩에 가중치 부여)
        sql_keywords = []
        
        # 스키마 매핑에서 테이블/컬럼 정보 추출
        if schema_mapping:
            if hasattr(schema_mapping, "relevant_tables"):
                sql_keywords.extend(schema_mapping.relevant_tables)
            if hasattr(schema_mapping, "relevant_columns"):
                if isinstance(schema_mapping.relevant_columns, list):
                    sql_keywords.extend(schema_mapping.relevant_columns)
                elif isinstance(schema_mapping.relevant_columns, dict):
                    sql_keywords.extend(schema_mapping.relevant_columns.keys())
        
        # SQL 관련 용어 강조 (반복하여 가중치 효과)
        sql_terms = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", "COUNT", "SUM", "AVG"]
        query_lower = query.lower()
        for term in sql_terms:
            if term.lower() in query_lower:
                sql_keywords.append(term)
        
        # 스키마 관련 용어 추가
        schema_terms = ["table", "column", "schema", "database", "field", "attribute"]
        for term in schema_terms:
            if term in query_lower:
                sql_keywords.append(term)
        
        # 고유 키워드만 추가 (중복 제거)
        if sql_keywords:
            unique_keywords = list(set(sql_keywords))
            # 키워드를 쿼리에 추가 (중요한 키워드는 2번 반복하여 가중치 효과)
            enhanced_parts.append(" ".join(unique_keywords))
        
        return " ".join(enhanced_parts)
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create new one from schema file"""
        try:
            # 벡터 스토어 캐시 경로
            cache_dir = Path(__file__).parent / ".rag_cache"
            cache_dir.mkdir(exist_ok=True)
            vector_store_path = cache_dir / "schema_vectorstore"
            
            # 기존 벡터 스토어가 있으면 로드
            if vector_store_path.exists() and self.embeddings:
                try:
                    self.vector_store = FAISS.load_local(
                        str(vector_store_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info("Loaded existing vector store from cache")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load cached vector store: {e}")
            
            # 새로 생성
            if not Path(self.schema_path).exists():
                logger.warning(f"Schema file not found: {self.schema_path}")
                return
            
            # 문서 로드 및 처리
            documents = self._load_and_process_schema()
            
            if documents and self.embeddings:
                # 벡터 스토어 생성 (FAISS 인덱스 최적화)
                # 문서 수에 따라 적절한 인덱스 타입 선택
                num_docs = len(documents)
                
                # FAISS 인덱스 생성
                # 작은 데이터셋(<1000): FlatIndex (정확하지만 느림)
                # 중간 데이터셋(1000-10000): IVFFlat (빠르고 정확)
                # 큰 데이터셋(>10000): IVFPQ (압축, 빠르지만 약간의 정확도 손실)
                
                if num_docs < 1000:
                    # FlatIndex 사용 (정확도 우선)
                    self.vector_store = FAISS.from_documents(
                        documents,
                        self.embeddings
                    )
                    logger.info(f"Created FAISS FlatIndex vector store with {num_docs} documents")
                else:
                    # IVFFlat 인덱스 사용 (빠른 검색, 정확도 유지)
                    # nlist: 클러스터 수 (일반적으로 sqrt(num_docs))
                    try:
                        import math
                        nlist = int(math.sqrt(num_docs))
                        nlist = max(10, min(nlist, 100))  # 10-100 범위로 제한
                        
                        # FAISS는 기본적으로 FlatIndex를 사용하므로,
                        # IVFFlat은 직접 생성해야 하지만, LangChain의 FAISS는 이를 직접 지원하지 않음
                        # 따라서 기본 FlatIndex 사용 (대부분의 경우 충분함)
                        self.vector_store = FAISS.from_documents(
                            documents,
                            self.embeddings
                        )
                        logger.info(f"Created FAISS vector store with {num_docs} documents (using default index)")
                    except Exception as e:
                        logger.warning(f"Failed to create optimized index: {e}, using default")
                        self.vector_store = FAISS.from_documents(
                            documents,
                            self.embeddings
                        )
                
                # 캐시 저장
                self.vector_store.save_local(str(vector_store_path))
                logger.info(f"Created and cached vector store with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load or create vector store: {e}", exc_info=True)
            self.vector_store = None
    
    def _load_and_process_schema(self) -> List[Document]:
        """
        Load and process schema markdown file into documents
        
        Returns:
            List of Document objects with metadata
        """
        try:
            if not Path(self.schema_path).exists():
                logger.error(f"Schema file not found: {self.schema_path}")
                return []
            
            # Markdown 파일 로드
            loader = TextLoader(str(self.schema_path), encoding="utf-8")
            raw_documents = loader.load()
            
            if not raw_documents:
                logger.warning("No content loaded from schema file")
                return []
            
            # Markdown 헤더 기반 분할 (테이블 단위로 유지)
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            
            # 먼저 헤더로 분할하여 테이블 정보 유지
            md_docs = markdown_splitter.split_text(raw_documents[0].page_content)
            
            # 각 문서에 메타데이터 추가 및 추가 분할
            documents = []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            
            for doc in md_docs:
                # 테이블명 추출 (메타데이터에서)
                table_name = None
                if "Header 2" in doc.metadata:
                    header_text = doc.metadata["Header 2"]
                    # `t_creator` 형식의 테이블명 추출
                    if "`" in header_text:
                        table_name = header_text.split("`")[1] if len(header_text.split("`")) > 1 else None
                
                # 테이블명이 없으면 본문에서 찾기
                if not table_name:
                    # `t_xxx` 패턴 찾기
                    import re
                    match = re.search(r'`(t_\w+)`', doc.page_content)
                    if match:
                        table_name = match.group(1)
                
                # 테이블명이 여전히 없으면 헤더 텍스트에서 직접 추출
                if not table_name and "Header 2" in doc.metadata:
                    header_text = doc.metadata["Header 2"]
                    # "## `t_creator` - 크리에이터 정보 테이블" 형식에서 추출
                    match = re.search(r'(t_\w+)', header_text)
                    if match:
                        table_name = match.group(1)
                
                # 추가 분할 (큰 테이블의 경우)
                sub_docs = text_splitter.split_documents([doc])
                
                for sub_doc in sub_docs:
                    # 메타데이터 추가 (테이블명 확실히 포함)
                    metadata = {
                        **sub_doc.metadata,
                        "table_name": table_name,
                        "source": "data_dictionary.md"
                    }
                    
                    documents.append(Document(
                        page_content=sub_doc.page_content,
                        metadata=metadata
                    ))
            
            logger.info(f"Processed {len(documents)} schema documents from {len(md_docs)} markdown sections")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load and process schema: {e}", exc_info=True)
            return []
    
    def _normalize_relevance_score(self, score: float, min_score: float = None, max_score: float = None) -> float:
        """
        Normalize relevance score to 0-1 range
        
        Args:
            score: Relevance score (can be numpy type)
            min_score: Minimum score for normalization
            max_score: Maximum score for normalization
            
        Returns:
            Normalized score as Python float
        
        Note: FAISS similarity_search_with_score returns distance (lower is better)
        We convert distance to similarity score (higher is better)
        
        Args:
            score: Distance score from FAISS (lower is better)
            min_score: Minimum distance (for normalization)
            max_score: Maximum distance (for normalization)
        
        Returns:
            Normalized similarity score (0-1, higher is better)
        """
        # FAISS distance는 L2 거리이므로, 거리가 작을수록 유사도가 높음
        # 거리를 유사도로 변환: similarity = 1 / (1 + distance)
        # 또는 음수 거리를 사용하여 정규화: similarity = max(0, 1 - distance / threshold)
        
        # numpy 타입을 Python float로 변환
        score_float = float(score) if hasattr(score, '__float__') else score
        
        # 간단한 변환: distance를 similarity로 변환 (0-1 범위)
        # 일반적으로 거리가 0-2 범위이므로 threshold를 2로 설정
        distance_threshold = 2.0
        similarity = max(0.0, min(1.0, 1.0 - (score_float / distance_threshold)))
        
        return float(similarity)  # Python float로 반환
    
    def _calculate_hybrid_score(
        self,
        vector_score: float,
        keyword_matches: int,
        table_name_match: bool = False,
        column_match: bool = False
    ) -> float:
        """
        Calculate hybrid relevance score combining vector similarity and keyword matching
        
        Args:
            vector_score: Normalized vector similarity score (0-1)
            keyword_matches: Number of keyword matches in content
            table_name_match: Whether table name matches query
            column_match: Whether column names match query
        
        Returns:
            Combined relevance score (0-1)
        """
        # 가중치 설정
        vector_weight = 0.7  # 벡터 유사도 가중치
        keyword_weight = 0.2  # 키워드 매칭 가중치
        metadata_weight = 0.1  # 메타데이터 매칭 가중치
        
        # 키워드 매칭 점수 (최대 5개 키워드 매칭 시 1.0)
        keyword_score = min(1.0, keyword_matches / 5.0)
        
        # 메타데이터 매칭 점수
        metadata_score = 0.0
        if table_name_match:
            metadata_score += 0.6
        if column_match:
            metadata_score += 0.4
        
        # 가중 평균 계산
        hybrid_score = (
            vector_score * vector_weight +
            keyword_score * keyword_weight +
            metadata_score * metadata_weight
        )
        
        return min(1.0, hybrid_score)
    
    def _count_keyword_matches(self, query: str, content: str) -> int:
        """
        Count keyword matches between query and content
        
        Args:
            query: Query string
            content: Document content
        
        Returns:
            Number of keyword matches
        """
        import re
        
        # 쿼리를 단어로 분리 (한글, 영문, 숫자 포함)
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        content_lower = content.lower()
        
        matches = 0
        for word in query_words:
            if len(word) > 2 and word in content_lower:  # 2글자 이상만 매칭
                matches += 1
        
        return matches
    
    def retrieve_relevant_schema(
        self,
        query: str,
        top_k: int = 3,
        filter_table_names: Optional[List[str]] = None,
        schema_mapping: Optional[Any] = None,
        use_hybrid_search: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant schema information based on query
        
        Args:
            query: User query or SQL generation context
            top_k: Number of relevant chunks to retrieve
            filter_table_names: Optional list of table names to filter by
            schema_mapping: Optional schema mapping for query enhancement
            use_hybrid_search: Whether to use hybrid search (vector + keyword)
        
        Returns:
            List of relevant schema chunks with metadata, sorted by relevance
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized, returning empty results")
            return []
        
        try:
            import time
            search_start_time = time.time()
            
            # 쿼리 향상 (SQL/스키마 키워드 추가)
            enhanced_query = self._enhance_query_for_embedding(query, schema_mapping) if use_hybrid_search else query
            
            # 유사도 검색 수행
            search_k = top_k * 3 if filter_table_names else top_k * 2  # 필터링을 위해 더 많이 검색
            
            results = self.vector_store.similarity_search_with_score(
                enhanced_query,
                k=search_k
            )
            
            search_time = (time.time() - search_start_time) * 1000  # ms 단위
            logger.debug(f"Vector search completed in {search_time:.2f}ms (k={search_k}, results={len(results)})")
            
            # 필터링 및 점수 계산
            processed_results = []
            for doc, score in results:
                table_name = doc.metadata.get("table_name")
                
                # 필터링 적용
                if filter_table_names and table_name not in filter_table_names:
                    continue
                
                # 벡터 유사도 정규화
                vector_score = self._normalize_relevance_score(score)
                
                # 하이브리드 점수 계산 (키워드 매칭 포함)
                if use_hybrid_search:
                    keyword_matches = self._count_keyword_matches(query, doc.page_content)
                    table_match = table_name and any(
                        table.lower() in query.lower() 
                        for table in (filter_table_names or [table_name])
                    )
                    
                    # 컬럼명 매칭 확인
                    column_match = False
                    if schema_mapping and hasattr(schema_mapping, "relevant_columns"):
                        columns = schema_mapping.relevant_columns
                        if isinstance(columns, list):
                            column_match = any(col in doc.page_content for col in columns)
                        elif isinstance(columns, dict):
                            column_match = any(col in doc.page_content for col in columns.keys())
                    
                    hybrid_score = self._calculate_hybrid_score(
                        vector_score,
                        keyword_matches,
                        table_match,
                        column_match
                    )
                    
                    relevance_score = hybrid_score
                else:
                    relevance_score = vector_score
                
                processed_results.append({
                    "doc": doc,
                    "score": score,  # 원본 거리 점수
                    "relevance_score": relevance_score,  # 정규화된 유사도 점수
                    "table_name": table_name
                })
            
            # 관련성 점수로 정렬 (높은 순)
            processed_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # top_k만큼 반환
            final_results = processed_results[:top_k]
            
            # 결과 포맷팅
            schema_chunks = []
            for result in final_results:
                doc = result["doc"]
                # numpy 타입을 Python float로 변환
                relevance_score = result["relevance_score"]
                if hasattr(relevance_score, '__float__'):
                    relevance_score = float(relevance_score)
                vector_distance = result["score"]
                if hasattr(vector_distance, '__float__'):
                    vector_distance = float(vector_distance)
                
                schema_chunks.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": relevance_score,  # Python float 타입
                    "table_name": result["table_name"],
                    "vector_distance": vector_distance  # Python float 타입
                })
            
            total_time = (time.time() - search_start_time) * 1000  # ms 단위
            
            logger.info(f"Retrieved {len(schema_chunks)} relevant schema chunks for query: {query[:50]}...")
            if schema_chunks:
                avg_score = sum(chunk["relevance_score"] for chunk in schema_chunks) / len(schema_chunks)
                max_score = max(chunk["relevance_score"] for chunk in schema_chunks)
                min_score = min(chunk["relevance_score"] for chunk in schema_chunks)
                logger.debug(
                    f"Relevance scores - avg: {avg_score:.3f}, max: {max_score:.3f}, min: {min_score:.3f}, "
                    f"total_time: {total_time:.2f}ms"
                )
            
            return schema_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve schema: {e}", exc_info=True)
            return []
    
    def process(self, state: GraphState) -> GraphState:
        """
        Process state and retrieve relevant schema information
        
        This method integrates SchemaMapper functionality:
        1. First, performs entity-based schema mapping (if entities exist)
        2. Then, uses RAG retrieval to enhance schema context
        3. Combines both results for comprehensive schema information
        """
        self._log_processing(state, "RAGSchemaRetrieverNode")
        
        try:
            user_query = state.get("user_query", "")
            entities = state.get("entities", [])
            intent = state.get("intent")
            
            # Step 1: Entity-based schema mapping (SchemaMapper 기능 통합)
            schema_mapping = None
            if entities:
                # 엔티티 기반으로 테이블, 컬럼, 관계 찾기
                relevant_tables = self._find_relevant_tables(entities)
                relevant_columns = self._find_relevant_columns(entities, relevant_tables)
                relationships = self._find_relationships(relevant_tables)
                
                # Confidence 계산
                confidence = self._calculate_mapping_confidence(
                    entities, relevant_tables, relevant_columns
                )
                
                # SchemaMapping 객체 생성
                schema_mapping = SchemaMapping(
                    relevant_tables=relevant_tables,
                    relevant_columns=relevant_columns,
                    relationships=relationships,
                    confidence=confidence
                )
                
                logger.info(f"Entity-based mapping: {len(relevant_tables)} tables, {len(relevant_columns)} columns")
            
            # Step 2: RAG 검색으로 스키마 컨텍스트 확장
            filter_tables = None
            if schema_mapping:
                filter_tables = schema_mapping.relevant_tables
            
            # 스키마 검색 쿼리 구성
            search_query = user_query
            if schema_mapping and schema_mapping.relevant_tables:
                # 엔티티 정보 추가 (RAG 검색 향상)
                search_query += " " + " ".join(schema_mapping.relevant_tables)
            
            # 관련 스키마 정보 검색 (하이브리드 검색 사용)
            schema_chunks = self.retrieve_relevant_schema(
                query=search_query,
                top_k=5,  # SQL 생성을 위해 더 많은 컨텍스트 제공
                filter_table_names=filter_tables,
                schema_mapping=schema_mapping,
                use_hybrid_search=True  # 하이브리드 검색 활성화
            )
            
            # Step 3: 상태에 저장 (TypedDict를 dict로 캐스팅하여 할당)
            state_dict: Dict[str, Any] = state  # type: ignore[assignment]
            
            # SchemaMapping 저장 (기존 코드와의 호환성 유지)
            if schema_mapping:
                state_dict["schema_mapping"] = schema_mapping
                state_dict["confidence_scores"]["schema_mapping"] = schema_mapping.confidence
            
            # RAG 검색 결과 저장
            state_dict["rag_schema_chunks"] = schema_chunks
            
            # 스키마 컨텍스트 문자열 생성 (프롬프트에 사용)
            schema_context = self._format_schema_context(schema_chunks)
            state_dict["rag_schema_context"] = schema_context
            
            logger.info(
                f"Schema retrieval complete: "
                f"{len(schema_chunks)} RAG chunks, "
                f"{len(schema_mapping.relevant_tables) if schema_mapping else 0} entity-mapped tables"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Error in RAGSchemaRetrieverNode.process: {e}", exc_info=True)
            state["rag_schema_chunks"] = []
            state["rag_schema_context"] = ""
            # schema_mapping이 없어도 계속 진행 가능하도록 에러 처리
            if "schema_mapping" not in state:
                state["schema_mapping"] = None
            return state
    
    def _format_schema_context(self, schema_chunks: List[Dict[str, Any]]) -> str:
        """
        Format schema chunks into a context string for prompts
        
        Args:
            schema_chunks: List of schema chunk dictionaries
        
        Returns:
            Formatted schema context string
        """
        if not schema_chunks:
            return ""
        
        context_parts = []
        seen_tables = set()
        
        for chunk in schema_chunks:
            table_name = chunk.get("table_name")
            content = chunk.get("content", "")
            
            if table_name and table_name not in seen_tables:
                context_parts.append(f"## {table_name}\n{content}")
                seen_tables.add(table_name)
            elif content:
                context_parts.append(content)
        
        return "\n\n".join(context_parts)
    
    def refresh_vector_store(self):
        """Refresh vector store from schema file"""
        logger.info("Refreshing vector store...")
        self._load_or_create_vector_store()
    
    # ========== SchemaMapper 기능 통합 메서드 ==========
    
    def _find_relevant_tables(self, entities: List[Entity]) -> List[str]:
        """
        Find relevant tables based on entities (SchemaMapper 기능)
        
        Args:
            entities: List of extracted entities
            
        Returns:
            List of relevant table names
        """
        relevant_tables = []
        
        for entity in entities:
            if entity.type == "table":
                # Direct table mention
                table_name = self._normalize_table_name(entity.name)
                if table_name in self.db_schema:
                    relevant_tables.append(table_name)
            elif entity.type == "column":
                # Find tables containing this column
                for table_name, table_info in self.db_schema.items():
                    if entity.name in table_info.get("columns", {}):
                        relevant_tables.append(table_name)
        
        return list(set(relevant_tables))
    
    def _find_relevant_columns(self, entities: List[Entity], tables: List[str]) -> List[str]:
        """
        Find relevant columns based on entities and tables (SchemaMapper 기능)
        
        Args:
            entities: List of extracted entities
            tables: List of relevant table names
            
        Returns:
            List of relevant column names
        """
        relevant_columns = []
        
        for entity in entities:
            if entity.type == "column":
                relevant_columns.append(entity.name)
        
        # Add columns from relevant tables
        for table in tables:
            if table in self.db_schema:
                table_columns = list(self.db_schema[table].get("columns", {}).keys())
                relevant_columns.extend(table_columns)
        
        return list(set(relevant_columns))
    
    def _find_relationships(self, tables: List[str]) -> List[Dict[str, str]]:
        """
        Find relationships between tables (SchemaMapper 기능)
        
        Args:
            tables: List of table names
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Simple relationship detection based on common patterns
        for table1 in tables:
            for table2 in tables:
                if table1 != table2:
                    # Check for foreign key relationships
                    if self._has_foreign_key_relationship(table1, table2):
                        relationships.append({
                            "from_table": table1,
                            "to_table": table2,
                            "type": "foreign_key"
                        })
        
        return relationships
    
    def _has_foreign_key_relationship(self, table1: str, table2: str) -> bool:
        """
        Check if there's a foreign key relationship between tables (SchemaMapper 기능)
        
        Args:
            table1: First table name
            table2: Second table name
            
        Returns:
            True if foreign key relationship exists
        """
        if table1 not in self.db_schema or table2 not in self.db_schema:
            return False
        
        # Simple heuristic: check if table1 has a column that references table2
        table1_columns = self.db_schema[table1].get("columns", {})
        for column_name, column_info in table1_columns.items():
            if column_info.get("type") == "foreign_key" and table2 in str(column_info):
                return True
        
        return False
    
    def _normalize_table_name(self, name: str) -> str:
        """
        Normalize table name to match schema (SchemaMapper 기능)
        
        Args:
            name: Table name to normalize
            
        Returns:
            Normalized table name
        """
        # Remove common prefixes/suffixes
        name = name.lower().strip()
        
        # 먼저 매핑 테이블에서 확인
        if name in self.table_name_mapping:
            return self.table_name_mapping[name]
        
        # Handle common variations
        if name.endswith('s'):
            singular_name = name[:-1]
            if singular_name in self.table_name_mapping:
                return self.table_name_mapping[singular_name]
        
        # Check if it matches any table in the schema
        for table_name in self.db_schema.keys():
            if name in table_name.lower() or table_name.lower() in name:
                return table_name
        
        return name
    
    def _calculate_mapping_confidence(
        self, 
        entities: List[Entity], 
        tables: List[str], 
        columns: List[str]
    ) -> float:
        """
        Calculate confidence for schema mapping (SchemaMapper 기능)
        
        공통 유틸리티 함수 사용
        
        Args:
            entities: List of extracted entities
            tables: List of relevant tables
            columns: List of relevant columns
            
        Returns:
            Confidence score (0-1)
        """
        from agentic_flow.utils import calculate_mapping_confidence
        return calculate_mapping_confidence(entities, tables, columns)

