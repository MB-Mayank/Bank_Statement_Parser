"""
Bank Statement Parser Agent - Autonomous code generation for PDF parsing
Uses LangGraph for agent orchestration with self-debugging capabilities
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, TypedDict
from dataclasses import dataclass
import re

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AgentConfig:
    """Agent configuration and paths"""
    target_bank: str
    data_dir: Path
    parser_dir: Path
    max_attempts: int = 3
    model: str = "llama-3.3-70b-versatile"
    
    @property
    def pdf_path(self) -> Path:
        return self.data_dir / self.target_bank / f"{self.target_bank}_sample.pdf"
    
    @property
    def csv_path(self) -> Path:
        return self.data_dir / self.target_bank / f"{self.target_bank}_expected.csv"
    
    @property
    def parser_path(self) -> Path:
        return self.parser_dir / f"{self.target_bank}_parser.py"


# ============================================================================
# State Management
# ============================================================================

class AgentState(TypedDict):
    """State shared across agent nodes"""
    config: AgentConfig
    messages: List
    current_code: Optional[str]
    test_result: Optional[str]
    attempt: int
    status: str  # planning, coding, testing, fixing, done, failed


# ============================================================================
# Agent Nodes
# ============================================================================

class ParserAgent:
    """Main agent orchestrator for parser generation"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY", "api_key"),
            model_name=config.model,
            temperature=0.1
        )
        
    def analyze_samples(self) -> Dict[str, str]:
        """Analyze PDF and CSV to understand parsing requirements"""
        print("📊 Analyzing sample data...")
        
        # Read expected CSV
        df = pd.read_csv(self.config.csv_path)
        
        # Get more detailed schema info
        csv_info = {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_rows": df.head(5).to_dict('records'),
            "shape": df.shape,
            "date_format_samples": {}
        }
        
        # Detect date columns and their formats
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_val = str(df[col].iloc[0]) if len(df) > 0 else ""
                if any(sep in sample_val for sep in ['-', '/', '.']):
                    csv_info["date_format_samples"][col] = sample_val
        
        print(f"✓ CSV Schema: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"✓ Columns: {df.columns.tolist()}")
        
        # Extract PDF text samples
        pdf_text = self._extract_pdf_text()
        
        return {
            "csv_schema": json.dumps(csv_info, indent=2, default=str),
            "pdf_sample": pdf_text,
            "bank": self.config.target_bank
        }
    
    def _extract_pdf_text(self) -> str:
        """Extract text from PDF using multiple methods"""
        try:
            import pdfplumber
            with pdfplumber.open(self.config.pdf_path) as pdf:
                # Extract first 2 pages
                texts = []
                for page in pdf.pages[:2]:
                    text = page.extract_text()
                    if text:
                        texts.append(text)
                    
                    # Also try to extract tables
                    tables = page.extract_tables()
                    if tables:
                        texts.append(f"\n[TABLE DETECTED: {len(tables)} tables on page]")
                
                return "\n---PAGE BREAK---\n".join(texts)[:2000]
        except Exception as e:
            return f"[PDF extraction failed: {e}]"
    
    def plan_node(self, state: AgentState) -> AgentState:
        """Planning phase: Analyze requirements and create strategy"""
        print("\n🎯 Planning phase...")
        
        analysis = self.analyze_samples()
        schema = json.loads(analysis['csv_schema'])
        
        planning_prompt = f"""You are an expert at parsing bank statement PDFs. Analyze and plan:

TARGET: {analysis['bank'].upper()} bank statement parser

EXPECTED OUTPUT SCHEMA:
- Columns: {schema['columns']}
- Data types: {schema['dtypes']}
- Expected rows: {schema['shape'][0]}
- Date columns: {schema.get('date_format_samples', {})}

PDF SAMPLE (first 2000 chars):
{analysis['pdf_sample'][:2000]}

PLANNING CHECKLIST:
1. PDF Library: Which is best? (pdfplumber for tables, PyPDF2 for text, tabula-py for complex tables)
2. Data Location: Where are transactions in the PDF? (table format, text lines, specific pages)
3. Column Mapping: How to extract each column from the PDF structure?
4. Date Parsing: What date format is used? How to parse it correctly?
5. Data Types: Which columns need float conversion? String cleaning?
6. Edge Cases: Missing values, multi-line entries, header/footer filtering

Provide a SPECIFIC 5-step extraction plan."""

        response = self.llm.invoke([SystemMessage(content=planning_prompt)])
        
        state["messages"].append(HumanMessage(content=planning_prompt))
        state["messages"].append(response)
        state["status"] = "coding"
        
        print(f"✅ Plan created")
        return state
    
    def code_node(self, state: AgentState) -> AgentState:
        """Code generation phase"""
        print("\n💻 Generating parser code...")
        
        analysis = self.analyze_samples()
        schema = json.loads(analysis['csv_schema'])
        
        # Build a detailed coding prompt with examples
        coding_prompt = f"""Write COMPLETE Python code for the bank statement parser.

REQUIREMENTS:
1. Function signature: def parse(pdf_path: str) -> pd.DataFrame
2. Output must have EXACTLY these columns: {schema['columns']}
3. Output must have EXACTLY {schema['shape'][0]} rows
4. Data types must match: {schema['dtypes']}
5. Sample expected row: {schema['sample_rows'][0] if schema['sample_rows'] else 'N/A'}

CRITICAL REQUIREMENTS:
- Use pdfplumber for PDF extraction
- Handle date parsing correctly: pd.to_datetime(..., errors='coerce') and convert back to string if object expected
- Convert numeric columns to float explicitly using pd.to_numeric(..., errors='coerce').fillna(0.0)
- Remove any header/footer rows
- Handle missing values with fillna('')
- Return a clean DataFrame with exact column order


EXAMPLE TEMPLATE:
```python
import pandas as pd
import pdfplumber
from typing import List, Dict

def parse(pdf_path: str) -> pd.DataFrame:
    '''Parse bank statement PDF and return DataFrame'''
    
    # Extract data from PDF
    transactions = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # METHOD 1: Try table extraction first
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    # Process table rows
                    pass
            
            # METHOD 2: Or extract text and parse lines
            text = page.extract_text()
            if text:
                lines = text.split('\\n')
                # Parse each line
                pass
    
    # Create DataFrame
    df = pd.DataFrame(transactions, columns={schema['columns']})
    

    # Fix type mismatches
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # convert invalids to NaT
        df['Date'] = df['Date'].dt.strftime('%d-%m-%Y')            # convert back to string if object expected

    for col in df.columns:
        if col.lower() in ['amount', 'balance', 'credit', 'debit']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    return df

```

Previous planning context:
{state['messages'][-1].content[:500] if state['messages'] else 'None'}

OUTPUT: Complete Python code ONLY. No explanations. Start with imports."""

        response = self.llm.invoke([
            SystemMessage(content="You are a Python expert. Output ONLY valid, complete Python code."),
            HumanMessage(content=coding_prompt)
        ])
        
        # Extract code
        code = self._extract_code(response.content)
        
        # Save code
        self.config.parser_path.parent.mkdir(exist_ok=True, parents=True)
        self.config.parser_path.write_text(code)
        
        state["current_code"] = code
        state["messages"].append(response)
        state["status"] = "testing"
        
        print(f"✅ Code generated ({len(code)} chars)")
        return state
    
    def test_node(self, state: AgentState) -> AgentState:
        """Testing phase: Run parser and compare with expected output"""
        print("\n🧪 Testing parser...")
        
        try:
            # Dynamic import
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "custom_parser", 
                self.config.parser_path
            )
            parser_module = importlib.util.module_from_spec(spec)
            sys.modules['custom_parser'] = parser_module
            spec.loader.exec_module(parser_module)
            
            # Run parser
            result_df = parser_module.parse(str(self.config.pdf_path))
            expected_df = pd.read_csv(self.config.csv_path)
            
            print(f"  Result shape: {result_df.shape}")
            print(f"  Expected shape: {expected_df.shape}")
            
            # Detailed comparison
            issues = self._compare_dataframes(result_df, expected_df)
            
            if not issues:
                state["test_result"] = "PASS"
                state["status"] = "done"
                print("✅ Tests passed!")
            else:
                state["test_result"] = " | ".join(issues)
                state["status"] = "fixing"
                print(f"❌ Tests failed:\n  " + "\n  ".join(issues))
                
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            state["test_result"] = f"RUNTIME ERROR: {str(e)}\n{error_detail}"
            state["status"] = "fixing"
            print(f"❌ Runtime error: {e}")
        
        state["attempt"] += 1
        return state
    
    # def test_node(self, state: AgentState) -> AgentState:
    #     """Testing phase: Run parser and compare with expected output"""
    #     print(f"\n🧪 Testing parser (attempt {state['attempt'] + 1}/{self.config.max_attempts})...")
        
    #     try:
    #         # Dynamic import of generated parser
    #         import importlib.util
    #         spec = importlib.util.spec_from_file_location(
    #             "custom_parser", 
    #             self.config.parser_path
    #         )
    #         parser_module = importlib.util.module_from_spec(spec)
    #         spec.loader.exec_module(parser_module)
            
    #         # Run parser
    #         result_df = parser_module.parse(str(self.config.pdf_path))
    #         expected_df = pd.read_csv(self.config.csv_path)
            
    #         # Detailed comparison
    #         comparison = self._compare_dataframes(result_df, expected_df)
            
    #         if comparison["match"]:
    #             state["test_result"] = "PASS"
    #             state["status"] = "done"
    #             print("✅ Tests passed!")
    #         else:
    #             error_msg = comparison["detailed_error"]
    #             state["test_result"] = f"FAIL: {error_msg}"
    #             state["error_history"] = state.get("error_history", []) + [error_msg]
                
    #             # Check attempts before deciding to fix
    #             state["attempt"] += 1
    #             if state["attempt"] >= self.config.max_attempts:
    #                 state["status"] = "failed"
    #                 print(f"❌ Max attempts ({self.config.max_attempts}) reached")
    #             else:
    #                 state["status"] = "fixing"
    #                 print(f"❌ Tests failed: {error_msg[:300]}...")
                
    #     except Exception as e:
    #         error_msg = f"EXECUTION ERROR: {type(e).__name__}: {str(e)}"
    #         state["test_result"] = error_msg
    #         state["error_history"] = state.get("error_history", []) + [error_msg]
            
    #         # Check attempts before deciding to fix
    #         state["attempt"] += 1
    #         if state["attempt"] >= self.config.max_attempts:
    #             state["status"] = "failed"
    #             print(f"❌ Max attempts ({self.config.max_attempts}) reached")
    #         else:
    #             state["status"] = "fixing"
    #             print(f"❌ Error: {e}")
        
    #     return state


    def fix_node(self, state: AgentState) -> AgentState:
        """Fixing phase: Debug and regenerate code"""
        
        if state["attempt"] >= self.config.max_attempts:
            state["status"] = "failed"
            print(f"\n❌ Max attempts ({self.config.max_attempts}) reached")
            return state
        
        print(f"\n🔧 Fixing attempt {state['attempt']}/{self.config.max_attempts}...")
        
        # Get expected schema for context
        expected_df = pd.read_csv(self.config.csv_path)
        schema_info = {
            "columns": expected_df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in expected_df.dtypes.items()},
            "sample_rows": expected_df.head(3).to_dict('records'),
            "row_count": len(expected_df)
        }
        
        fix_prompt = f"""DEBUG AND FIX the parser code. It has errors.

CURRENT CODE:
```python
{state['current_code'][:3000]}
```

ERROR/ISSUES:
{state['test_result'][:1000]}

EXPECTED SCHEMA:
{json.dumps(schema_info, indent=2, default=str)}

DEBUGGING CHECKLIST:
1. If "could not convert string to float" → Check date parsing (use pd.to_datetime)
2. If "Shape mismatch" → Verify PDF extraction logic, check page iteration
3. If empty DataFrame → Print debug info, check extraction method
4. If column mismatch → Verify column names match exactly
5. If type mismatch → Add explicit type conversions

COMMON FIXES:
- Date parsing: pd.to_datetime(df['Date'], format='%d-%m-%Y') or infer_datetime_format=True
- Amount parsing: pd.to_numeric(df['Amount'], errors='coerce')
- Empty results: Check if using correct extraction method (tables vs text)
- Row filtering: Remove header/footer rows, check for empty strings

OUTPUT: Complete CORRECTED Python code only. Fix the specific issues mentioned."""

        response = self.llm.invoke([
            SystemMessage(content="You are debugging Python code. Output ONLY the corrected complete code."),
            HumanMessage(content=fix_prompt)
        ])
        
        # Extract and save
        code = self._extract_code(response.content)
        self.config.parser_path.write_text(code)
        
        state["current_code"] = code
        state["messages"].append(response)
        state["status"] = "testing"
        
        return state
    
    # def _extract_code(self, content: str) -> str:
    #     """Extract Python code from LLM response"""
    #     # Try multiple extraction methods
    #     if "```python" in content:
    #         parts = content.split("```python")
    #         if len(parts) > 1:
    #             code = parts[1].split("```")[0]
    #             return code.strip()
        
    #     if "```" in content:
    #         parts = content.split("```")
    #         if len(parts) >= 3:
    #             code = parts[1]
    #             return code.strip()
        
    #     # Fallback: assume entire content is code
    #     return content.strip()
    
    

    def _extract_code(self, content: str) -> str:
        """Extract Python code from LLM response using regex."""
        # Match ```python ... ``` or ``` ... ```
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        matches = re.findall(pattern, content, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
        # Fallback: assume entire content is code
        return content.strip()




    def _compare_dataframes(self, result: pd.DataFrame, expected: pd.DataFrame) -> List[str]:
        """Generate detailed comparison of DataFrames"""
        issues = []
        
        # Shape check
        if result.shape != expected.shape:
            issues.append(f"Shape mismatch: got {result.shape}, expected {expected.shape}")
            if result.shape[0] == 0:
                issues.append("  → Result DataFrame is EMPTY! Check PDF extraction logic.")
            return issues  # Critical error, return early
        
        # Column check
        if list(result.columns) != list(expected.columns):
            issues.append(f"Column mismatch:\n    Got: {list(result.columns)}\n    Expected: {list(expected.columns)}")
        
        # Type check
        for col in expected.columns:
            if col in result.columns:
                if result[col].dtype != expected[col].dtype:
                    issues.append(
                        f"Type mismatch in '{col}': got {result[col].dtype}, expected {expected[col].dtype}"
                        f"\n    Sample values: {result[col].head(2).tolist()}"
                    )
        
        # Value check (if shapes match)
        if not issues:
            try:
                pd.testing.assert_frame_equal(result, expected, check_dtype=True)
            except AssertionError as e:
                issues.append(f"Value differences: {str(e)[:200]}")
        
        return issues
    
    def should_continue(self, state: AgentState) -> str:
        status = state["status"]
        
        if status == "done":
            return "end"
        elif status == "failed":
            return "end"
        elif status == "fixing":
            return "fix"
        
        return "end"


# ============================================================================
# Graph Construction
# ============================================================================

def build_agent_graph(config: AgentConfig) -> StateGraph:
    """Build LangGraph workflow"""
    agent = ParserAgent(config)
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("plan", agent.plan_node)
    workflow.add_node("code", agent.code_node)
    workflow.add_node("test", agent.test_node)
    workflow.add_node("fix", agent.fix_node)
    
    # Define flow
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "code")
    workflow.add_edge("code", "test")
    # workflow.add_edge("fix", "test")
    
    # Conditional routing from test
    workflow.add_conditional_edges(
        "test",
        agent.should_continue,
        {
            # "done": END,
            # "failed": END,
            # "fixing": "fix"
            "fix": "fix",
            "end": END
        }
    )
    workflow.add_edge("fix", "test")
    return workflow.compile()


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bank Statement Parser Agent - Generate custom parsers using LLM"
    )
    parser.add_argument("--target", required=True, help="Target bank (e.g., icici, sbi)")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--parser-dir", default="custom_parsers", help="Parser output directory")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max fixing attempts")
    
    args = parser.parse_args()
    
    # Setup config
    config = AgentConfig(
        target_bank=args.target,
        data_dir=Path(args.data_dir),
        parser_dir=Path(args.parser_dir),
        max_attempts=args.max_attempts
    )
    
    # Validate inputs
    if not config.pdf_path.exists():
        print(f"❌ Error: PDF not found at {config.pdf_path}")
        sys.exit(1)
    
    if not config.csv_path.exists():
        print(f"❌ Error: CSV not found at {config.csv_path}")
        sys.exit(1)
    
    print("="*60)
    print(f"🚀 Bank Statement Parser Agent")
    print("="*60)
    print(f"Target: {args.target.upper()}")
    print(f"PDF: {config.pdf_path}")
    print(f"CSV: {config.csv_path}")
    print(f"Output: {config.parser_path}")
    print(f"Max attempts: {config.max_attempts}")
    print("="*60)
    
    # Initialize state
    initial_state = AgentState(
        config=config,
        messages=[],
        current_code=None,
        test_result=None,
        attempt=0,
        status="planning"
    )
    
    # Run agent
    try:
        graph = build_agent_graph(config)
        final_state = graph.invoke(initial_state)
        
        # Report results
        print("\n" + "="*60)
        if final_state["status"] == "done":
            print("✅ SUCCESS! Parser generated and validated.")
            print(f"📄 Parser: {config.parser_path}")
            print(f"🧪 Tests: PASSED")
            print(f"🔄 Attempts: {final_state['attempt']}")
        else:
            print("❌ FAILED to generate working parser")
            print(f"🔄 Attempts: {final_state['attempt']}/{config.max_attempts}")
            print(f"📋 Last error:\n{final_state.get('test_result', 'Unknown')[:500]}")
            sys.exit(1)
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Agent crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()