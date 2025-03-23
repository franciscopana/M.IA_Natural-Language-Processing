#!/bin/bash

# Convert all Jupyter notebooks to Python scripts
jupyter nbconvert --to script *.ipynb

# Process each Python file to remove image data lines and combine into notebook.py
> notebook.py  # Create/clear the notebook.py file

for file in *.py; do
    # Skip notebook.py itself to prevent duplication
    if [ "$file" != "notebook.py" ]; then
        echo "# Content from $file" >> notebook.py
        echo "" >> notebook.py
        
        sed '/data:image\//d' "$file" >> notebook.py
        
        echo -e "\n\n# ================================================\n" >> notebook.py
    fi
done

# Copy content to clipboard for Ctrl+V functionality
if command -v xclip > /dev/null; then
    # For Linux
    cat notebook.py | xclip -selection clipboard
    echo "Content copied to clipboard. You can now use Ctrl+V to paste."
elif command -v pbcopy > /dev/null; then
    # For macOS
    cat notebook.py | pbcopy
    echo "Content copied to clipboard. You can now use Ctrl+V to paste."
else
    echo "Unable to copy to clipboard. Please install xclip (Linux), pbcopy (macOS), or clip.exe (Windows)."
fi