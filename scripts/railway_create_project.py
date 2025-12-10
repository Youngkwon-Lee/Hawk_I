"""Railway project creation script - continues from logged in state"""
from playwright.sync_api import sync_playwright
import time

def create_railway_project():
    with sync_playwright() as p:
        # Launch browser (visible)
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Go to Railway new project page
        print("1. Going to Railway new project page...")
        page.goto('https://railway.app/new')
        page.wait_for_load_state('networkidle')
        time.sleep(2)

        page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)
        print("   Screenshot saved")

        # Click on "Deploy from GitHub repo"
        print("2. Looking for 'Deploy from GitHub repo'...")
        try:
            # Wait for the page to fully load
            page.wait_for_selector('text=Deploy from GitHub repo', timeout=10000)
            github_button = page.locator('text=Deploy from GitHub repo').first
            github_button.click()
            print("   Clicked 'Deploy from GitHub repo'")
            page.wait_for_load_state('networkidle')
            time.sleep(2)
        except Exception as e:
            print(f"   Error: {e}")

        page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)

        # Search for Hawk_I repo
        print("3. Searching for Hawk_I repository...")
        try:
            # Look for search input
            search_input = page.locator('input[placeholder*="Search"]').first
            if search_input.is_visible():
                search_input.fill('Hawk_I')
                time.sleep(2)
                print("   Searched for Hawk_I")
        except Exception as e:
            print(f"   Search error: {e}")

        page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)

        # Click on Hawk_I repo
        print("4. Selecting Hawk_I repository...")
        time.sleep(2)
        try:
            hawk_repo = page.locator('text=Hawk_I').first
            if hawk_repo.is_visible():
                hawk_repo.click()
                print("   Selected Hawk_I")
                page.wait_for_load_state('networkidle')
                time.sleep(2)
        except Exception as e:
            print(f"   Selection error: {e}")

        page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)

        # Configure deployment
        print("5. Configuring deployment...")
        time.sleep(2)

        # Look for "Add Root Directory" or similar
        try:
            # Try to find root directory option
            root_option = page.locator('text=Root Directory').first
            if root_option.is_visible():
                root_option.click()
                time.sleep(1)

            # Try to find input and set to 'backend'
            root_input = page.locator('input').filter(has_text='').first
            inputs = page.locator('input').all()
            for inp in inputs:
                placeholder = inp.get_attribute('placeholder') or ''
                if 'root' in placeholder.lower() or 'directory' in placeholder.lower():
                    inp.fill('backend')
                    print("   Set root directory to 'backend'")
                    break
        except Exception as e:
            print(f"   Config error: {e}")

        page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)

        # Try to deploy
        print("6. Looking for Deploy button...")
        time.sleep(2)
        try:
            deploy_btn = page.locator('button:has-text("Deploy")').first
            if deploy_btn.is_visible():
                deploy_btn.click()
                print("   Clicked Deploy!")
                page.wait_for_load_state('networkidle')
                time.sleep(5)
        except Exception as e:
            print(f"   Deploy error: {e}")

        page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)

        print("\n=== Current State ===")
        print(f"URL: {page.url}")
        print("Screenshot saved to railway_state.png")
        print("\nKeeping browser open for 60 seconds...")

        # Keep browser open for a bit to see result
        time.sleep(60)
        page.screenshot(path='C:/Users/YK/tulip/Hawkeye/railway_state.png', full_page=True)

        browser.close()
        print("Done!")

if __name__ == "__main__":
    create_railway_project()
