#!/bin/bash
# Simple monitoring script for the pipeline

echo "════════════════════════════════════════════════════════════════"
echo "🔍 TABERTA Pipeline Monitor"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Tmux session: taberta_pipeline"
echo "Status: $(tmux has-session -t taberta_pipeline 2>/dev/null && echo '✅ Running' || echo '❌ Not found')"
echo ""
echo "📁 Log file: pipeline_execution.log"
echo "Size: $(ls -lh pipeline_execution.log 2>/dev/null | awk '{print $5}' || echo 'N/A')"
echo ""
echo "📈 Latest progress (last 30 lines):"
echo "────────────────────────────────────────────────────────────────"
tail -30 pipeline_execution.log 2>/dev/null | grep -E "Successfully loaded|Tables loaded|Loading dataset|STEP|Error|✅|❌" || tail -30 pipeline_execution.log
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Commands:"
echo "  • Attach:   tmux attach -t taberta_pipeline"
echo "  • Log:      tail -f pipeline_execution.log"
echo "  • Detach:   Ctrl+B then D"
echo "════════════════════════════════════════════════════════════════"
