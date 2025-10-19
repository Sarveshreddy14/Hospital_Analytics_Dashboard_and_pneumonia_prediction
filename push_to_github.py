#!/usr/bin/env python3
"""
Quick GitHub Push Script
Usage: python push_to_github.py [username] [repository_name]
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
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python push_to_github.py <username> [repository_name]")
        print("Example: python push_to_github.py myusername hospital-analytics")
        return
    
    username = sys.argv[1]
    repo_name = sys.argv[2] if len(sys.argv) > 2 else "hospital-analytics"
    
    print(f"🚀 Pushing to GitHub: {username}/{repo_name}")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("comprehensive_dashboard.py").exists():
        print("❌ Error: Please run from project root directory")
        return
    
    # Add and commit changes
    run_command("git add .", "Adding files")
    run_command('git commit -m "Update Hospital Analytics Dashboard"', "Committing changes")
    
    # Set up remote and push
    remote_url = f"https://github.com/{username}/{repo_name}.git"
    
    # Remove existing remote if it exists
    subprocess.run("git remote remove origin", shell=True, capture_output=True)
    
    if run_command(f"git remote add origin {remote_url}", "Adding remote origin"):
        if run_command("git branch -M main", "Setting main branch"):
            if run_command("git push -u origin main", "Pushing to GitHub"):
                print(f"\n🎉 Successfully pushed to GitHub!")
                print(f"🔗 Repository: https://github.com/{username}/{repo_name}")
            else:
                print("\n❌ Push failed. Make sure the repository exists on GitHub.")
        else:
            print("\n❌ Failed to set main branch")
    else:
        print("\n❌ Failed to add remote origin")

if __name__ == "__main__":
    main()
