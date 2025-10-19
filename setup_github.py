#!/usr/bin/env python3
"""
GitHub Setup Script for Hospital Analytics Project
This script helps you push your project to GitHub
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_git_installed():
    """Check if git is installed"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_git_config():
    """Check if git is configured"""
    try:
        subprocess.run(["git", "config", "user.name"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main setup function"""
    print("🏥 Hospital Analytics - GitHub Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("comprehensive_dashboard.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   (The directory containing comprehensive_dashboard.py)")
        return
    
    # Check git installation
    if not check_git_installed():
        print("❌ Git is not installed. Please install Git first:")
        print("   https://git-scm.com/downloads")
        return
    
    # Check git configuration
    if not check_git_config():
        print("⚠️  Git is not configured. Please set up your identity:")
        print("   git config --global user.name 'Your Name'")
        print("   git config --global user.email 'your.email@example.com'")
        return
    
    print("✅ Git is properly configured")
    
    # Initialize git repository
    if not Path(".git").exists():
        if not run_command("git init", "Initializing Git repository"):
            return
    else:
        print("✅ Git repository already initialized")
    
    # Add all files
    if not run_command("git add .", "Adding files to Git"):
        return
    
    # Check if there are changes to commit
    result = subprocess.run("git diff --cached --quiet", shell=True, capture_output=True)
    if result.returncode == 0:
        print("ℹ️  No changes to commit")
    else:
        # Commit changes
        if not run_command('git commit -m "Initial commit: Hospital Analytics Dashboard"', "Committing changes"):
            return
    
    print("\n🎉 Local Git repository is ready!")
    print("\n📋 Next steps to push to GitHub:")
    print("1. Go to https://github.com and create a new repository")
    print("2. Name it 'hospital-analytics' (or your preferred name)")
    print("3. Don't initialize with README (we already have one)")
    print("4. Copy the repository URL")
    print("5. Run the following commands:")
    print()
    print("   git remote add origin https://github.com/YOUR_USERNAME/hospital-analytics.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    print()
    print("🔗 Or run: python push_to_github.py YOUR_USERNAME REPO_NAME")
    
    # Ask if user wants to continue with push
    response = input("\n❓ Do you want to continue with pushing to GitHub now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        username = input("Enter your GitHub username: ").strip()
        repo_name = input("Enter repository name (default: hospital-analytics): ").strip() or "hospital-analytics"
        
        # Add remote origin
        remote_url = f"https://github.com/{username}/{repo_name}.git"
        if run_command(f"git remote add origin {remote_url}", "Adding remote origin"):
            # Push to GitHub
            if run_command("git branch -M main", "Renaming branch to main"):
                if run_command("git push -u origin main", "Pushing to GitHub"):
                    print(f"\n🎉 Successfully pushed to GitHub!")
                    print(f"🔗 Repository URL: https://github.com/{username}/{repo_name}")
                else:
                    print("\n❌ Failed to push to GitHub. Please check your repository URL and try again.")
        else:
            print("\n❌ Failed to add remote origin. Please check your repository URL.")

if __name__ == "__main__":
    main()
