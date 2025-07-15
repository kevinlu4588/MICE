#!/bin/bash
echo "Pipeline Results"
echo "================"
echo ""
echo "Configuration:"
cat config.txt
echo ""
echo "Logs:"
ls -la logs/
echo ""
if [ -d "evaluation" ]; then
    echo "Evaluation Results:"
    ls -la evaluation/ | head -20
fi
