"""
SQL Generation and Validation Components

This module contains the SQLGenerator and SQLValidator nodes for the pipeline.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from .state import (
    GraphState, SchemaMapping, SQLResult, 
    QueryIntent, QueryComplexity, Entity
)
from .nodes import BaseNode
from core.config import get_settings
from core.db import get_db_session
from core.logging import get_logger

logger = get_logger(__name__)


class SQLGenerator(BaseNode):
    """SQL generation node."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = self._initialize_llm()
        self.db_schema = config.get("db_schema", {})
        
        # 테이블 이름 매핑: 일반적인 이름 -> 실제 DB 테이블명
        self.table_name_mapping = {
            "users": "t_member",
            "user": "t_member",
            "members": "t_member",
            "member": "t_member",
            "회원": "t_member",
            "사용자": "t_member",
            "user_info": "t_member_info",
            "member_info": "t_member_info",
            "회원정보": "t_member_info",
            "user_profile": "t_member_profile",
            "member_profile": "t_member_profile",
            "profiles": "t_member_profile",
            "프로필": "t_member_profile",
            "creators": "t_creator",
            "creator": "t_creator",
            "크리에이터": "t_creator",
            "창작자": "t_creator",
            "fundings": "t_funding",
            "funding": "t_funding",
            "펀딩": "t_funding",
            "projects": "t_funding",
            "프로젝트": "t_funding",
            "funding_members": "t_funding_member",
            "backers": "t_funding_member",
            "supporters": "t_funding_member",
            "후원자": "t_funding_member",
            "follows": "t_follow",
            "follow": "t_follow",
            "팔로우": "t_follow",
            "orders": "t_order",
            "order": "t_order",
            "주문": "t_order",
        }
    
    def _initialize_llm(self):
        """Initialize the LLM for SQL generation."""
        settings = get_settings()
        try:
            return ChatGoogleGenerativeAI(
                model=settings.llm.model,
                google_api_key=settings.llm.api_key,
                temperature=0.1,  # Low temperature for consistent SQL generation
                max_output_tokens=2048
            )
        except Exception as e:
            error_msg = str(e)
            if "credentials" in error_msg.lower():
                self.logger.warning("LLM credentials not configured. Using mock LLM for testing.")
            elif "api_key" in error_msg.lower():
                self.logger.warning("LLM API key not found. Using mock LLM for testing.")
            else:
                self.logger.warning(f"Failed to initialize LLM: {error_msg}. Using mock LLM.")
            # Try to create a simple LLM instance for testing
            try:
                import os
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    google_api_key=os.environ.get('GOOGLE_API_KEY', ''),
                    temperature=0.1,
                    max_output_tokens=1024,
                    convert_system_message_to_human=True  # 최신 버전 호환성
                )
            except:
                return None
    
    def process(self, state: GraphState) -> GraphState:
        """Generate SQL query from processed natural language."""
        self._log_processing(state, "SQLGenerator")
        
        try:
            normalized_query = state.get("normalized_query", "")
            intent = state.get("intent")
            entities = state.get("entities", [])
            schema_mapping = state.get("schema_mapping")
            
            if not schema_mapping:
                state["error_message"] = "Schema mapping required for SQL generation"
                return state
            
            # Generate SQL
            sql_query = self._generate_sql(
                normalized_query, intent, entities, schema_mapping
            )
            
            # Validate and optimize SQL
            validated_sql = self._validate_sql_syntax(sql_query)
            optimized_sql = self._optimize_sql(validated_sql, schema_mapping)
            
            # Calculate confidence and complexity
            confidence = self._calculate_sql_confidence(sql_query, schema_mapping)
            complexity = self._assess_sql_complexity(optimized_sql)
            
            # Create SQL result
            sql_result = SQLResult(
                sql_query=optimized_sql,
                confidence=confidence,
                complexity=complexity
            )
            
            state["sql_result"] = sql_result
            state["confidence_scores"]["sql_generation"] = confidence
            
            self.logger.info(f"Generated SQL: {optimized_sql[:100]}...")
            self.logger.info(f"Confidence: {confidence}, Complexity: {complexity}")
            
        except Exception as e:
            self.logger.error(f"Error in SQLGenerator: {str(e)}")
            state["error_message"] = f"SQL generation failed: {str(e)}"
        
        return state
    
    def _generate_sql(self, query: str, intent: QueryIntent, entities: List[Entity], schema_mapping: SchemaMapping) -> str:
        """Generate SQL query using LLM."""
        
        # Create schema context
        schema_context = self._build_schema_context(schema_mapping)
        
        # Create few-shot examples
        examples = self._get_few_shot_examples()
        
        # 실제 테이블 이름 리스트 생성
        actual_tables = list(self.db_schema.keys())
        table_list = ", ".join(actual_tables)
        
        system_prompt = f"""
        You are a SQL expert. Generate a SQL query based on the natural language request.
        
        **IMPORTANT: Use ONLY these exact table names in your SQL:**
        Available Tables: {table_list}
        
        Common mistakes to avoid:
        - DO NOT use "users" - use "t_member" instead
        - DO NOT use "fundings" - use "t_funding" instead
        - DO NOT use "creators" - use "t_creator" instead
        
        Database Schema:
        {schema_context}
        
        Examples:
        {examples}
        
        Rules:
        1. Only use SELECT statements (read-only access)
        2. **MUST use exact table names from the "Available Tables" list above**
        3. Use proper table and column names from the schema
        4. Include appropriate WHERE clauses for filtering
        5. Use JOINs when multiple tables are needed
        6. Add LIMIT clauses for large result sets
        7. Use proper SQL syntax for MariaDB/MySQL
        """
        
        try:
            if self.llm is None:
                # Mock SQL for testing when LLM is not available
                if "회원" in query or "사용자" in query:
                    return "SELECT email, nickname FROM t_member LIMIT 1000;"
                elif "크리에이터" in query:
                    return "SELECT nickname FROM t_creator LIMIT 1000;"
                elif "펀딩" in query:
                    return "SELECT title, status FROM t_funding LIMIT 1000;"
                else:
                    return "SELECT 1 as placeholder;"
            
            # 최신 LangChain 방식: SystemMessage 대신 HumanMessage에 시스템 프롬프트 포함
            messages = [
                HumanMessage(content=f"{system_prompt}\n\nGenerate SQL for: {query}")
            ]
            
            response = self.llm.invoke(messages)
            sql_query = self._extract_sql_from_response(response.content)
            
            return sql_query
            
        except Exception as e:
            self.logger.error(f"Error generating SQL: {str(e)}")
            return "SELECT 1 as error"
    
    def _build_schema_context(self, schema_mapping: SchemaMapping) -> str:
        """Build schema context string for LLM."""
        context_parts = []
        
        for table in schema_mapping.relevant_tables:
            if table in self.db_schema:
                table_info = self.db_schema[table]
                columns = table_info.get("columns", {})
                
                column_list = []
                for col_name, col_info in columns.items():
                    col_type = col_info.get("type", "unknown")
                    col_nullable = "NULL" if col_info.get("nullable", True) else "NOT NULL"
                    column_list.append(f"  {col_name} ({col_type}) {col_nullable}")
                
                context_parts.append(f"Table {table}:\n" + "\n".join(column_list))
        
        return "\n\n".join(context_parts)
    
    def _get_few_shot_examples(self) -> str:
        """Get few-shot examples for SQL generation."""
        return """
        Example 1:
        Query: "모든 회원의 이메일 주소를 보여줘"
        SQL: SELECT email FROM t_member;
        
        Example 2:
        Query: "활성 상태인 회원 수를 세어줘"
        SQL: SELECT COUNT(*) FROM t_member WHERE status = 'A';
        
        Example 3:
        Query: "크리에이터의 닉네임을 보여줘"
        SQL: SELECT nickname FROM t_creator;
        
        Example 4:
        Query: "진행 중인 펀딩 제목을 보여줘"
        SQL: SELECT title FROM t_funding WHERE status != 'fail';
        """
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        # Look for SQL code blocks
        sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Look for SQL without code blocks
        sql_match = re.search(r'(SELECT.*?;?)', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Fallback
        return "SELECT 1 as error"
    
    def _validate_sql_syntax(self, sql: str) -> str:
        """Validate and clean SQL syntax."""
        # Remove extra whitespace
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # Ensure it starts with SELECT
        if not sql.upper().startswith('SELECT'):
            sql = f"SELECT {sql}"
        
        # Fix MariaDB reserved words and common issues
        sql = self._fix_mariadb_syntax(sql)
        
        # Add semicolon if missing
        if not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def _fix_mariadb_syntax(self, sql: str) -> str:
        """Fix MariaDB-specific syntax issues."""
        # Replace problematic column aliases
        problematic_aliases = {
            'as exists': 'as table_exists',
            'as exists': 'as table_exists',
            'as limit': 'as limit_count',
            'as order': 'as order_value',
            'as group': 'as group_value',
        }
        
        for old_alias, new_alias in problematic_aliases.items():
            sql = re.sub(old_alias, new_alias, sql, flags=re.IGNORECASE)
        
        # Fix SHOW statements to use INFORMATION_SCHEMA
        if sql.upper().startswith('SHOW TABLES'):
            sql = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = DATABASE()"
            if 'LIKE' in sql.upper():
                like_pattern = re.search(r"LIKE\s+'([^']+)'", sql, re.IGNORECASE)
                if like_pattern:
                    pattern = like_pattern.group(1)
                    sql = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME LIKE '{pattern}'"
        
        return sql
    
    def _optimize_sql(self, sql: str, schema_mapping: SchemaMapping) -> str:
        """Optimize SQL query."""
        # 테이블 이름을 실제 DB 테이블명으로 교체
        original_sql = sql
        sql = self._replace_table_names(sql)
        
        # 디버깅: 교체 전후 비교
        if original_sql != sql:
            self.logger.info(f"Table name replacement: '{original_sql}' -> '{sql}'")
        
        # Add LIMIT clause for large result sets
        if 'LIMIT' not in sql.upper():
            sql = sql.rstrip(';') + ' LIMIT 1000;'
        
        # Ensure proper table references
        for table in schema_mapping.relevant_tables:
            if table not in sql.upper():
                # Add table reference if missing
                pass  # This would need more sophisticated logic
        
        return sql
    
    def _replace_table_names(self, sql: str) -> str:
        """Replace generic table names with actual DB table names."""
        # 테이블 이름을 가장 긴 것부터 정렬 (예: "users"가 "user"보다 먼저 매칭되도록)
        sorted_mappings = sorted(
            self.table_name_mapping.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        # SQL에서 테이블 이름을 찾아 교체
        for generic_name, actual_name in sorted_mappings:
            # FROM, JOIN 절에서 테이블 이름 교체 (대소문자 무시, 단어 경계 고려)
            # 패턴: FROM/JOIN 다음에 오는 테이블 이름
            patterns = [
                (r'\bFROM\s+' + re.escape(generic_name) + r'\b', f'FROM {actual_name}'),
                (r'\bJOIN\s+' + re.escape(generic_name) + r'\b', f'JOIN {actual_name}'),
                # 일반 단어 경계 교체 (마지막 폴백)
                (r'\b' + re.escape(generic_name) + r'\b', actual_name),
            ]
            
            for pattern, replacement in patterns:
                sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
        
        return sql
    
    def _calculate_sql_confidence(self, sql: str, schema_mapping: SchemaMapping) -> float:
        """Calculate confidence for generated SQL."""
        base_confidence = 0.7
        
        # Check if SQL uses relevant tables
        sql_upper = sql.upper()
        used_tables = [table for table in schema_mapping.relevant_tables if table.upper() in sql_upper]
        
        if used_tables:
            base_confidence += 0.2
        
        # Check for proper SQL structure
        if 'SELECT' in sql_upper and 'FROM' in sql_upper:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _assess_sql_complexity(self, sql: str) -> QueryComplexity:
        """Assess the complexity of the SQL query."""
        sql_upper = sql.upper()
        
        # Count complexity indicators
        complexity_score = 0
        
        if 'JOIN' in sql_upper:
            complexity_score += 2
        if 'GROUP BY' in sql_upper:
            complexity_score += 1
        if 'HAVING' in sql_upper:
            complexity_score += 1
        if 'ORDER BY' in sql_upper:
            complexity_score += 1
        if 'LIMIT' in sql_upper:
            complexity_score += 1
        
        # Determine complexity level
        if complexity_score <= 1:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 3:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.COMPLEX


class SQLValidator(BaseNode):
    """SQL validation node."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def process(self, state: GraphState) -> GraphState:
        """Validate the generated SQL query."""
        self._log_processing(state, "SQLValidator")
        
        try:
            sql_result = state.get("sql_result")
            if not sql_result:
                state["error_message"] = "No SQL result to validate"
                return state
            
            sql_query = sql_result.sql_query
            
            # Perform validation checks
            is_valid, error_message = self._validate_sql(sql_query)
            
            if is_valid:
                state["is_valid"] = True
                state["final_sql"] = sql_query
                state["success"] = True
                self.logger.info("SQL validation successful")
            else:
                state["is_valid"] = False
                state["error_message"] = error_message
                state["retry_count"] = state.get("retry_count", 0) + 1
                self.logger.warning(f"SQL validation failed: {error_message}")
            
        except Exception as e:
            self.logger.error(f"Error in SQLValidator: {str(e)}")
            state["error_message"] = f"SQL validation failed: {str(e)}"
            state["is_valid"] = False
        
        return state
    
    def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL query for security and correctness."""
        
        # Security checks
        security_checks = [
            (self._check_read_only, "Query must be read-only"),
            (self._check_no_dangerous_operations, "No dangerous operations allowed"),
            (self._check_syntax_valid, "Invalid SQL syntax")
        ]
        
        for check_func, error_msg in security_checks:
            is_valid, error = check_func(sql)
            if not is_valid:
                return False, f"{error_msg}: {error}"
        
        return True, None
    
    def _check_read_only(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Check if query is read-only."""
        sql_upper = sql.upper()
        
        # Block DML operations
        dangerous_operations = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
        
        for operation in dangerous_operations:
            if operation in sql_upper:
                return False, f"Blocked {operation} operation"
        
        # Must start with SELECT
        if not sql_upper.strip().startswith('SELECT'):
            return False, "Query must start with SELECT"
        
        return True, None
    
    def _check_no_dangerous_operations(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Check for dangerous operations."""
        sql_upper = sql.upper()
        
        # Block system operations
        dangerous_patterns = [
            'INTO OUTFILE',
            'LOAD_FILE',
            'INTO DUMPFILE',
            'EXECUTE',
            'PREPARE',
            'SHOW PROCESSLIST'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in sql_upper:
                return False, f"Blocked dangerous pattern: {pattern}"
        
        return True, None
    
    def _check_syntax_valid(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Check basic SQL syntax validity."""
        try:
            # Basic syntax checks
            if not sql.strip():
                return False, "Empty query"
            
            # Check for balanced parentheses
            if sql.count('(') != sql.count(')'):
                return False, "Unbalanced parentheses"
            
            # Check for basic SELECT structure
            if 'SELECT' not in sql.upper():
                return False, "Missing SELECT clause"
            
            return True, None
            
        except Exception as e:
            return False, f"Syntax check failed: {str(e)}"