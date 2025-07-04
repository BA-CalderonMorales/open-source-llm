#!/usr/bin/env python3
"""
Bridge between Python inference capabilities and Rust MVP.

This script demonstrates how to integrate the existing Python inference
stack with our new Rust-based conversation interface, setting up for 
Phase 2 optimizations described in the PRD.
"""

import os
import sys
import subprocess
import sqlite3
from pathlib import Path

def check_rust_mvp():
    """Check if our Rust MVP is built and working."""
    mvp_path = Path("inference-re/target/debug/chat")
    if not mvp_path.exists():
        print("❌ Rust MVP not found. Building...")
        result = subprocess.run([
            "cargo", "build", "--bin", "chat", 
            "--manifest-path", "inference-re/Cargo.toml"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to build Rust MVP: {result.stderr}")
            return False
        print("✅ Rust MVP built successfully")
    else:
        print("✅ Rust MVP already built")
    return True

def inspect_conversation_db():
    """Inspect the SQLite database created by our Rust MVP."""
    db_path = "conversation_history.db"
    if not os.path.exists(db_path):
        print("📝 No conversation history found. The database will be created on first use.")
        return
    
    print(f"📊 Inspecting conversation database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table schema
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        schema = cursor.fetchone()
        if schema:
            print(f"🏗️  Database schema:")
            print(f"   {schema[0]}")
        
        # Get conversation count
        cursor.execute("SELECT COUNT(*) FROM conversations")
        count = cursor.fetchone()[0]
        print(f"💬 Total conversations: {count}")
        
        if count > 0:
            # Show recent conversations
            cursor.execute("""
                SELECT timestamp, user_message, assistant_response, token_count 
                FROM conversations 
                ORDER BY id DESC 
                LIMIT 3
            """)
            
            print("🕒 Recent conversations:")
            for row in cursor.fetchall():
                timestamp, user_msg, assistant_resp, tokens = row
                print(f"   [{timestamp[:19]}] Tokens: {tokens}")
                print(f"   User: {user_msg[:50]}{'...' if len(user_msg) > 50 else ''}")
                print(f"   Assistant: {assistant_resp[:50]}{'...' if len(assistant_resp) > 50 else ''}")
                print()
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")

def demonstrate_integration():
    """Show how Python and Rust components can work together."""
    print("\n🔗 Demonstrating Python-Rust Integration")
    print("=" * 50)
    
    # Check if Python dependencies are available
    try:
        import torch
        print("✅ PyTorch available for Python inference")
    except ImportError:
        print("⚠️  PyTorch not available (optional for this demo)")
    
    # Show the integration architecture
    print("\n🏗️  Current Architecture:")
    print("   📱 Rust MVP: Chat interface + SQLite context storage")
    print("   🐍 Python: Advanced inference capabilities (Triton kernels)")
    print("   🔄 Integration: Share data via SQLite and file I/O")
    
    print("\n📋 Phase 2 Roadmap:")
    print("   1. Rust tokenization (fast text processing)")
    print("   2. Rust context retrieval (vector similarity search)")
    print("   3. Python-Rust model bridge (hybrid inference)")
    print("   4. Performance benchmarking and optimization")

def run_mvp_demo():
    """Run a quick demo of the MVP to show it's working."""
    print("\n🚀 Running MVP Demo")
    print("=" * 30)
    
    try:
        # Run the chat app with a simple test
        result = subprocess.run([
            "./inference-re/target/debug/chat"
        ], input="!stats\nquit\n", text=True, capture_output=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ MVP demo completed successfully")
            # Show the output
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():  # Skip empty lines
                    print(f"   {line}")
        else:
            print(f"❌ MVP demo failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⚠️  MVP demo timed out (this is normal)")
    except Exception as e:
        print(f"❌ Error running MVP demo: {e}")

def main():
    """Main integration demonstration."""
    print("🤖 DeepSeek Local-First Integration Demo")
    print("=" * 60)
    print("This script demonstrates the connection between our Rust MVP")
    print("and existing Python capabilities, preparing for Phase 2.\n")
    
    # Step 1: Verify Rust MVP is ready
    if not check_rust_mvp():
        sys.exit(1)
    
    # Step 2: Inspect any existing conversation data
    inspect_conversation_db()
    
    # Step 3: Show integration possibilities
    demonstrate_integration()
    
    # Step 4: Quick demo
    run_mvp_demo()
    
    print("\n✨ Integration demo complete!")
    print("\n🚀 Next Steps for Phase 2:")
    print("   • Implement Rust tokenizer for faster text processing")
    print("   • Add vector embeddings for smarter context retrieval")
    print("   • Create Python-Rust bridge for hybrid inference")
    print("   • Benchmark performance improvements")
    print("\n💡 Try running: ./inference-re/target/debug/chat")

if __name__ == "__main__":
    main()