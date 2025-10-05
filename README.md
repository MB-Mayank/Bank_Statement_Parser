# Bank Statement Parser Agent Architecture

## 🔄 Flow Chart

```
START
  ↓
┌─────────┐
│  PLAN   │ → Analyze PDF + CSV schema
│  Node   │ → Create extraction strategy
└────┬────┘
     ↓
┌─────────┐
│  CODE   │ → Generate parse() function
│  Node   │ → Save to {bank}_parser.py
└────┬────┘
     ↓
┌─────────┐
│  TEST   │ → Import + Execute parser
│  Node   │ → Compare result vs expected
└────┬────┘
     ↓
  Decision?
     ├─→ PASS ────────────→ END (done)
     ├─→ attempt >= 3 ────→ END (failed)
     └─→ FAIL ↓
          ┌─────────┐
          │   FIX   │ → Debug with error context
          │  Node   │ → Regenerate code
          └────┬────┘
               ↓
          (loop back to TEST)
```

## 🏗️ Node Architecture

### **PLAN Node**
```
Input:  PDF text (2000 chars), CSV schema
Action: LLM analyzes structure → creates extraction plan
Output: Planning strategy → status="coding"
```

### **CODE Node**
```
Input:  Plan context, expected schema
Action: LLM generates complete parse() function
Output: Python file saved → status="testing"
```

### **TEST Node**
```
Input:  Generated parser, sample PDF, expected CSV
Action: Dynamic import → execute → compare DataFrames
        - Shape check (rows × cols)
        - Column names check
        - Data types check
        - Values check
Output: "PASS" or error details
        attempt += 1
        
Route:  PASS → status="done" → END
        attempt >= max_attempts → status="failed" → END
        FAIL → status="fixing" → FIX
```

### **FIX Node**
```
Input:  Current code, error message, expected schema
Action: LLM debugs with error context → regenerates code
Output: Corrected Python file → status="testing" → TEST
```

## 📊 State Schema

```python
AgentState = {
    config: AgentConfig          # Paths (pdf/csv/parser)
    messages: List               # LLM conversation
    current_code: str           # Latest generated code
    test_result: str            # "PASS" or error details
    attempt: int                # 0 to max_attempts
    status: str                 # "planning"|"coding"|"testing"|"fixing"|"done"|"failed"
}
```

## 🎯 Key Components

**AgentConfig**:
```python
target_bank: "icici"
pdf_path: data/icici/icici_sample.pdf
csv_path: data/icici/icici_expected.csv
parser_path: custom_parsers/icici_parser.py
max_attempts: 3
```

**ParserAgent Methods**:
- `analyze_samples()` - Extract PDF text + CSV schema
- `_extract_code()` - Regex extraction from LLM response
- `_compare_dataframes()` - Detailed validation logic
- `should_continue()` - Routing decision (end/fix)

## 🔁 Self-Healing Loop

```
TEST ⟷ FIX (max 3 iterations)
  ↓
 END
```

Each iteration:
1. TEST finds specific error (shape/type/value mismatch)
2. FIX receives error + code + expected schema
3. LLM regenerates corrected version
4. Loop until PASS or max_attempts

## 🛠️ Graph Construction

```python
workflow = StateGraph(AgentState)
workflow.add_node("plan", agent.plan_node)
workflow.add_node("code", agent.code_node)
workflow.add_node("test", agent.test_node)
workflow.add_node("fix", agent.fix_node)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "code")
workflow.add_edge("code", "test")
workflow.add_conditional_edges("test", agent.should_continue, {"fix": "fix", "end": END})
workflow.add_edge("fix", "test")
```

**Linear Flow**: PLAN → CODE → TEST  
**Conditional Loop**: TEST ⟷ FIX (until done/failed)
