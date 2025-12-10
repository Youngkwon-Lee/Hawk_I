"""Railway deployment automation script"""
from playwright.sync_api import sync_playwright
import time

def deploy_to_railway():
    with sync_playwright() as p:
        # Launch browser (visible for user to interact if needed)
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Go to Railway
        print("1. Navigating to Railway...")
        page.goto('https://railway.app/new')
        page.wait_for_load_state('networkidle')

        # Take screenshot to see current state
        page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)
        print("Screenshot saved to railway_state.png")

        # Check if logged in by looking for "New Project" or login button
        page_content = page.content()

        if 'Sign in' in page_content or 'Log in' in page_content or 'Login' in page_content:
            print("2. Not logged in. Please log in manually in the browser...")
            print("   Waiting for login...")

            # Wait for user to log in (max 2 minutes)
            for i in range(24):  # 24 * 5 = 120 seconds
                time.sleep(5)
                page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)
                current_url = page.url
                content = page.content()

                if 'dashboard' in current_url or 'new' in current_url:
                    if 'Deploy from GitHub' in content or 'GitHub Repo' in content:
                        print("   Login successful!")
                        break
                print(f"   Waiting... ({(i+1)*5}s)")

        # Now try to deploy from GitHub
        print("3. Looking for GitHub deploy option...")
        page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)

        # Try to find and click "Deploy from GitHub repo"
        try:
            github_button = page.locator('text=Deploy from GitHub').first
            if github_button.is_visible():
                github_button.click()
                page.wait_for_load_state('networkidle')
                print("   Clicked 'Deploy from GitHub'")
        except:
            print("   Could not find GitHub deploy button")

        page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)

        # Look for Hawk_I repository
        print("4. Looking for Hawk_I repository...")
        time.sleep(2)

        try:
            # Search or find the repo
            hawk_repo = page.locator('text=Hawk_I').first
            if hawk_repo.is_visible():
                hawk_repo.click()
                page.wait_for_load_state('networkidle')
                print("   Selected Hawk_I repository")
        except:
            print("   Could not find Hawk_I repository")

        page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)

        # Configure root directory
        print("5. Configuring root directory to 'backend'...")
        time.sleep(2)

        try:
            # Look for root directory configuration
            root_dir_input = page.locator('input[placeholder*="root"]').first
            if root_dir_input.is_visible():
                root_dir_input.fill('backend')
                print("   Set root directory to 'backend'")
        except:
            print("   Could not find root directory input")

        page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)

        print("\n=== Current State ===")
        print(f"URL: {page.url}")
        print("Screenshot saved. Please check railway_state.png")
        print("\nKeeping browser open for manual completion if needed...")
        print("Press Ctrl+C to close when done.")

        # Keep browser open
        try:
            while True:
                time.sleep(10)
                page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)
        except KeyboardInterrupt:
            pass
        finally:
            browser.close()

if __name__ == "__main__":
    deploy_to_railway()
