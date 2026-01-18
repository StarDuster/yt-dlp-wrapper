"""
YouTube browser authentication using Playwright.

Uses persistent browser context to maintain login state across sessions.
Exports cookies in Netscape format for yt-dlp.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from playwright.sync_api import BrowserContext, sync_playwright

from .. import config

logger = logging.getLogger(__name__)


class YouTubeBrowserAuth:
    """
    Manage YouTube authentication using Playwright persistent browser context.
    
    This class handles:
    - Browser login with persistent profile (retains refresh tokens)
    - Cookie export in Netscape format for yt-dlp
    - Login state detection
    """
    
    YOUTUBE_URL = "https://www.youtube.com"
    YOUTUBE_LOGIN_URL = "https://accounts.google.com/ServiceLogin?service=youtube"
    
    def __init__(
        self,
        profile_dir: Optional[Path] = None,
        cookies_file: Optional[Path] = None,
    ):
        """
        Initialize YouTube browser auth.
        
        Args:
            profile_dir: Directory to store browser profile (persistent state)
            cookies_file: Path to export Netscape format cookies
        """
        self.profile_dir = profile_dir or config.YOUTUBE_BROWSER_PROFILE
        self.cookies_file = cookies_file or config.YOUTUBE_COOKIES_FILE
        
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.cookies_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger
    
    def login(self, headless: bool = False, timeout: int = 300) -> bool:
        """
        Launch browser for user to login to YouTube.
        
        Opens a persistent browser context, navigates to YouTube, and waits
        for user to complete login. The browser state (including refresh tokens)
        is automatically saved to the profile directory.
        
        Args:
            headless: If True, run in headless mode (not useful for manual login)
            timeout: Maximum seconds to wait for login (default: 5 minutes)
            
        Returns:
            True if login successful, False otherwise
        """
        self.logger.info(f"Starting YouTube login browser...")
        self.logger.info(f"Browser profile: {self.profile_dir}")
        self.logger.info(f"Cookies file: {self.cookies_file}")
        
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=headless,
                viewport={"width": 1280, "height": 800},
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
            )
            
            try:
                page = context.new_page()
                
                page.goto(self.YOUTUBE_URL, wait_until="domcontentloaded")
                time.sleep(2)
                
                if self._is_logged_in(page):
                    self.logger.info("Already logged in to YouTube!")
                    self._export_cookies(context)
                    print("\n✓ Already logged in! Cookies exported.")
                    return True
                
                self.logger.info("Not logged in. Navigating to login page...")
                page.goto(self.YOUTUBE_LOGIN_URL, wait_until="domcontentloaded")
                
                print("\n" + "=" * 60)
                print("YouTube Login Required")
                print("=" * 60)
                print("A browser window should have opened.")
                print("Please complete the login process in the browser.")
                print("Once logged in, you can close the browser window.")
                print(f"Waiting up to {timeout} seconds for login...")
                print("=" * 60 + "\n")
                
                start_time = time.time()
                check_interval = 3
                redirected_after_google_login = False
                
                while time.time() - start_time < timeout:
                    try:
                        time.sleep(check_interval)
                        
                        if not page.is_closed():
                            current_url = page.url

                            if "youtube.com" in current_url:
                                redirected_after_google_login = False
                                if self._is_logged_in(page):
                                    self.logger.info("Login detected!")
                                    self._export_cookies(context)
                                    print("\n✓ Login successful! Cookies exported.")
                                    print(f"  Cookies saved to: {self.cookies_file}")
                                    return True
                            else:
                                google_logged_in = False
                                try:
                                    for cookie in page.context.cookies():
                                        if (
                                            cookie.get("name") in ["SID", "SSID", "HSID"]
                                            and "google.com" in (cookie.get("domain") or "")
                                        ):
                                            google_logged_in = True
                                            break
                                except Exception:
                                    google_logged_in = False

                                if google_logged_in and not redirected_after_google_login:
                                    redirected_after_google_login = True
                                    print(
                                        "\n\n✓ Detected Google login cookies (SID/SSID/HSID). Redirecting to YouTube to verify login state..."
                                    )
                                    try:
                                        page.goto(self.YOUTUBE_URL, wait_until="domcontentloaded")
                                        time.sleep(2)
                                    except Exception as nav_err:
                                        self.logger.warning(f"Failed to navigate back to YouTube: {nav_err}")
                                        redirected_after_google_login = False

                                if "youtube.com" in page.url and self._is_logged_in(page):
                                    self.logger.info("Login detected!")
                                    self._export_cookies(context)
                                    print("\n✓ Login successful! Cookies exported.")
                                    print(f"  Cookies saved to: {self.cookies_file}")
                                    return True
                        else:
                            raise Exception("Page closed")
                            
                    except Exception as e:
                        print(f"\nBrowser closed or disconnected ({e}). Verifying login state...")
                        break
                    
                    elapsed = int(time.time() - start_time)
                    remaining = timeout - elapsed
                    print(f"\r  Waiting for login... ({remaining}s remaining)", end="", flush=True)

            except Exception as e:
                self.logger.warning(f"Browser interaction error: {e}")
            finally:
                try:
                    context.close()
                except Exception:
                    pass
        
        print("\nVerifying login status with fresh instance...")
        return self.refresh_cookies()
    
    def _is_logged_in(self, page) -> bool:
        """
        Check if the user is logged in to YouTube.
        
        Looks for the avatar button which only appears when logged in.
        """
        try:
            selectors = [
                'button#avatar-btn',
                'yt-img-shadow#avatar-btn',
                'a[href*="/channel/"]',
                '#avatar-btn',
            ]
            
            for selector in selectors:
                try:
                    element = page.query_selector(selector)
                    if element:
                        return True
                except Exception:
                    continue
            
            signin_selectors = [
                'a[href*="accounts.google.com/ServiceLogin"]',
                'ytd-button-renderer a[href*="accounts.google.com"]',
                'tp-yt-paper-button[aria-label="Sign in"]',
            ]
            
            for selector in signin_selectors:
                try:
                    element = page.query_selector(selector)
                    if element and element.is_visible():
                        return False
                except Exception:
                    continue
            
            cookies = page.context.cookies()
            for cookie in cookies:
                if cookie["name"] in ["SID", "SSID", "HSID"] and "google.com" in cookie["domain"]:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking login status: {e}")
            return False
    
    def _export_cookies(self, context: BrowserContext) -> None:
        """
        Export cookies from browser context to Netscape format file.
        
        The Netscape format is what yt-dlp expects with --cookies option.
        """
        cookies = context.cookies()
        
        relevant_domains = [".youtube.com", ".google.com", "youtube.com", "google.com"]
        filtered_cookies = [
            c for c in cookies
            if any(domain in c["domain"] for domain in relevant_domains)
        ]
        
        self.logger.info(f"Exporting {len(filtered_cookies)} cookies to {self.cookies_file}")
        
        with open(self.cookies_file, "w") as f:
            f.write("# Netscape HTTP Cookie File\n")
            f.write("# https://curl.haxx.se/docs/http-cookies.html\n")
            f.write("# This file was generated by yt-dlp-wrapper browser_auth\n\n")
            
            for cookie in filtered_cookies:
                domain = cookie["domain"]
                flag = "TRUE" if domain.startswith(".") else "FALSE"
                path = cookie.get("path", "/")
                secure = "TRUE" if cookie.get("secure", False) else "FALSE"
                expiry = int(cookie.get("expires", -1))
                if expiry < 0:
                    expiry = 0
                name = cookie["name"]
                value = cookie["value"]
                
                f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")
        
        self.logger.info(f"Cookies exported successfully")
    
    def refresh_cookies(self) -> bool:
        """
        Refresh cookies by launching browser with existing profile.
        
        This loads the persistent profile (which may have valid refresh tokens)
        and exports fresh cookies.
        
        Returns:
            True if cookies refreshed successfully, False otherwise
        """
        if not self.profile_dir.exists():
            self.logger.error("No browser profile found. Please run login first.")
            return False
        
        self.logger.info("Refreshing cookies from existing profile...")
        
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
            )
            
            try:
                page = context.new_page()
                page.goto(self.YOUTUBE_URL, wait_until="domcontentloaded")
                time.sleep(3)
                
                if self._is_logged_in(page):
                    self._export_cookies(context)
                    self.logger.info("Cookies refreshed successfully")
                    return True
                else:
                    self.logger.warning("Not logged in. Profile may have expired.")
                    return False
                    
            finally:
                context.close()
    
    def is_logged_in(self) -> bool:
        """
        Check if we have a valid login session.
        
        Returns:
            True if logged in, False otherwise
        """
        if not self.profile_dir.exists():
            return False
        
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=True,
                args=["--no-sandbox"],
            )
            
            try:
                page = context.new_page()
                page.goto(self.YOUTUBE_URL, wait_until="domcontentloaded")
                time.sleep(2)
                return self._is_logged_in(page)
            finally:
                context.close()
    
    def cookies_exist(self) -> bool:
        """Check if cookies file exists."""
        return self.cookies_file.exists()
    
    def get_cookies_file(self) -> Optional[Path]:
        """
        Get path to cookies file if it exists.
        
        Returns:
            Path to cookies file, or None if not found
        """
        if self.cookies_file.exists():
            return self.cookies_file
        return None

    def clear_auth(self) -> bool:
        """
        Clear all authentication data (browser profile and cookies).
        
        Use this to switch to a different YouTube account.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        import shutil
        
        cleared_something = False
        
        if self.cookies_file.exists():
            try:
                self.cookies_file.unlink()
                self.logger.info(f"Deleted cookies file: {self.cookies_file}")
                cleared_something = True
            except Exception as e:
                self.logger.error(f"Failed to delete cookies file: {e}")
                return False
        
        if self.profile_dir.exists():
            try:
                shutil.rmtree(self.profile_dir)
                self.logger.info(f"Deleted browser profile: {self.profile_dir}")
                cleared_something = True
            except Exception as e:
                self.logger.error(f"Failed to delete browser profile: {e}")
                return False
        
        if cleared_something:
            self.logger.info("Authentication data cleared. You can now login with a different account.")
        else:
            self.logger.info("No authentication data to clear.")
        
        return True


def youtube_clear_auth() -> bool:
    """
    Convenience function to clear YouTube authentication data.
    
    Use this to switch to a different YouTube account.
    
    Returns:
        True if cleared successfully
    """
    auth = YouTubeBrowserAuth()
    return auth.clear_auth()


def youtube_browser_login(headless: bool = False) -> bool:
    """
    Convenience function to run YouTube browser login.
    
    Args:
        headless: If True, run in headless mode
        
    Returns:
        True if login successful
    """
    auth = YouTubeBrowserAuth()
    return auth.login(headless=headless)


def youtube_refresh_cookies() -> bool:
    """
    Convenience function to refresh YouTube cookies.
    
    Returns:
        True if refresh successful
    """
    auth = YouTubeBrowserAuth()
    return auth.refresh_cookies()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    auth = YouTubeBrowserAuth()
    print(f"Profile dir: {auth.profile_dir}")
    print(f"Cookies file: {auth.cookies_file}")
    print(f"Cookies exist: {auth.cookies_exist()}")
    
    success = auth.login()
    print(f"Login result: {success}")
