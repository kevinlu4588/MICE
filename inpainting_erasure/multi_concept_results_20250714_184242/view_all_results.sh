#!/bin/bash
echo "Multi-Concept Pipeline Results"
echo "=============================="
echo ""
echo "Summary:"
cat all_concepts_summary.log | tail -20
echo ""
echo "Detailed Results:"
cat results_summary.csv | column -t -s','
echo ""
echo "Individual concept results:"
for dir in */; do
    if [ -d "$dir" ] && [ "$dir" != "*/" ]; then
        echo "  - $dir"
    fi
done
echo ""
echo "To view a specific concept's results:"
echo "  cd CONCEPT_NAME"
