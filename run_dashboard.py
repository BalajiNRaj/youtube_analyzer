#!/usr/bin/env python3
"""
Launcher script for the YouTube Analytics Streamlit Dashboard
"""

import subprocess
import sys
import os
import webbrowser
import time

def main():
    """Launch the Streamlit application."""
    print("🚀 Starting YouTube Analytics Dashboard...")
    print("📊 This will open in your default web browser")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Change to the correct directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Check if requirements are installed
        try:
            import streamlit
            import plotly
            print("✅ Dependencies verified")
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("\n💡 Installing missing requirements...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Launch Streamlit
        print("🔄 Starting Streamlit server...")
        print("📍 URL: http://localhost:8501")
        print("=" * 60)
        
        # Auto-open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open('http://localhost:8501')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        cmd = [
            sys.executable, 
            "-m", 
            "streamlit", 
            "run", 
            "streamlit_app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--theme.primaryColor=#FF4B4B",
            "--theme.backgroundColor=#FFFFFF",
            "--theme.secondaryBackgroundColor=#F0F2F6"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down dashboard...")
        print("👋 Thanks for using YouTube Analytics Dashboard!")
    except Exception as e:
        print(f"❌ Error starting dashboard: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you have installed the requirements:")
        print("      pip install -r requirements.txt")
        print("   2. Check if port 8501 is available")
        print("   3. Ensure you're in the correct directory")

if __name__ == "__main__":
    main()
