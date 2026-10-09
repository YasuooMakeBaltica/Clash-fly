@echo off
REM Double-click to edit the fly's decks in your browser.
cd /d "%~dp0"
python -m flybrain.deck_editor
pause
