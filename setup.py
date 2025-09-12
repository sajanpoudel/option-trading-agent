#!/usr/bin/env python3
"""
Neural Options Oracle++ Setup Script
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def check_docker():
    """Check if Docker is available"""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("✅ Docker is available")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Docker not found (optional for development)")
        return False

def setup_environment():
    """Setup environment file"""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from .env.example")
        print("⚠️  Please configure your API keys in .env file")
        return True
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("❌ .env.example file not found")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["logs", "tests/unit", "tests/integration", "tests/e2e"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")
    return True

def install_dependencies():
    """Install Python dependencies"""
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    return True

def test_basic_setup():
    """Test basic setup"""
    try:
        # Test importing main modules
        sys.path.insert(0, str(Path.cwd()))
        from config.settings import settings
        from config.logging import setup_logging
        
        print("✅ Basic imports working")
        return True
    except Exception as e:
        print(f"❌ Basic setup test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Neural Options Oracle++ Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    # Optional checks
    check_docker()
    
    # Setup steps
    steps = [
        setup_environment,
        create_directories,
        install_dependencies,
        test_basic_setup
    ]
    
    for step in steps:
        if not step():
            print(f"\n❌ Setup failed at step: {step.__name__}")
            sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Configure your API keys in .env file:")
    print("   - SUPABASE_URL and SUPABASE_SERVICE_KEY")
    print("   - OPENAI_API_KEY")
    print("   - GEMINI_API_KEY")
    print("   - JIGSAWSTACK_API_KEY")
    print("   - ALPACA_API_KEY and ALPACA_SECRET_KEY")
    print("\n2. Run the application:")
    print("   python main.py")
    print("\n3. Or use Docker:")
    print("   docker-compose up")
    print("\n4. Access the API documentation:")
    print("   http://localhost:8080/docs")

if __name__ == "__main__":
    main()