#!/usr/bin/env python3
"""
Screenshot Script for Summit Housing Dashboard
Takes high-quality screenshots of the static dashboard for portfolio use.
Supports both Light and Dark modes.
"""

import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright


# Configuration
BASE_URL = "http://localhost:8000"
OUTPUT_DIR = Path(__file__).parent.parent / "screenshots"
VIEWPORT_DESKTOP = {"width": 1920, "height": 1080}
VIEWPORT_MOBILE = {"width": 430, "height": 932}  # iPhone 14 Pro Max


async def set_theme(page, theme):
    """Force the application theme (light/dark)."""
    # 1. Update localStorage so it persists if we reload (though we don't here)
    # 2. Set the data-theme attribute which drives the CSS
    # 3. Wait for the CSS transition (0.3s)
    print(f"  🎨 Setting theme to: {theme}")
    await page.evaluate(f"""
        localStorage.setItem('theme', '{theme}');
        document.documentElement.setAttribute('data-theme', '{theme}');
    """)
    await page.wait_for_timeout(500)  # Wait for transition


async def mobile_navigate(page, target_page, wait_time=2000):
    """Navigate on mobile by opening menu, clicking, and closing menu."""
    # Open the mobile menu
    await page.click('#menu-toggle')
    await page.wait_for_timeout(500)
    # Click the navigation item
    await page.click(f'button[data-page="{target_page}"]')
    await page.wait_for_timeout(wait_time)


async def capture_desktop(browser, mode="light"):
    """Capture desktop screenshots in specified mode."""
    suffix = "_dark" if mode == "dark" else ""
    print(f"\n🖥️  Taking desktop screenshots ({mode} mode)...")
    
    context = await browser.new_context(
        viewport=VIEWPORT_DESKTOP,
        device_scale_factor=2,  # Retina quality
        color_scheme=mode
    )
    page = await context.new_page()
    
    # Navigate to the dashboard
    await page.goto(BASE_URL, wait_until="networkidle")
    print("✅ Page loaded")
    
    # Force the correct theme
    await set_theme(page, mode)

    # Wait for initial content to render
    await page.wait_for_timeout(2000)
    
    # 1. Introduction Page
    print("  📷 Capturing: Introduction page...")
    await page.click('button[data-page="intro"]')
    await page.wait_for_timeout(1000)
    await page.screenshot(
        path=OUTPUT_DIR / f"01_intro_desktop{suffix}.png",
        full_page=True
    )
    
    # 2. Data Story Page - Landscape
    print("  📷 Capturing: Data Story - Landscape...")
    await page.click('button[data-page="story"]')
    await page.wait_for_timeout(2000)  # Wait for map to load
    await page.screenshot(
        path=OUTPUT_DIR / f"02_story_landscape_desktop{suffix}.png",
        full_page=False
    )
    
    # 3. Data Story - Ski Lift Effect
    print("  📷 Capturing: Data Story - Ski Lift Effect...")
    # Scroll to activate the ski lift step
    await page.evaluate("""
        document.querySelector('[data-step="ski-lift"]').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    """)
    await page.wait_for_timeout(2000)
    await page.screenshot(
        path=OUTPUT_DIR / f"03_story_ski_lift_desktop{suffix}.png",
        full_page=False
    )
    
    # 4. Data Story - Market Cycles
    print("  📷 Capturing: Data Story - Market Cycles...")
    await page.evaluate("""
        document.querySelector('[data-step="market-cycles"]').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    """)
    await page.wait_for_timeout(2000)
    await page.screenshot(
        path=OUTPUT_DIR / f"04_story_market_cycles_desktop{suffix}.png",
        full_page=False
    )
    
    # 5. Data Story - Buyer Origins
    print("  📷 Capturing: Data Story - Buyer Origins...")
    await page.evaluate("""
        document.querySelector('[data-step="buyer-origins"]').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    """)
    await page.wait_for_timeout(2000)
    await page.screenshot(
        path=OUTPUT_DIR / f"05_story_buyer_origins_desktop{suffix}.png",
        full_page=False
    )
    
    # 6. Data Story - Seasonality
    print("  📷 Capturing: Data Story - Seasonality...")
    await page.evaluate("""
        document.querySelector('[data-step="seasonality"]').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    """)
    await page.wait_for_timeout(2000)
    await page.screenshot(
        path=OUTPUT_DIR / f"06_story_seasonality_desktop{suffix}.png",
        full_page=False
    )
    
    # 7. ML Experiments Page
    print("  📷 Capturing: ML Experiments - Leaderboard...")
    await page.click('button[data-page="experiments"]')
    await page.wait_for_timeout(2000)
    await page.screenshot(
        path=OUTPUT_DIR / f"07_experiments_leaderboard_desktop{suffix}.png",
        full_page=True
    )
    
    # 8. ML Experiments - Model Detail (click first model)
    print("  📷 Capturing: ML Experiments - Model Detail...")
    # Click the first "View" button in the leaderboard
    view_button = await page.query_selector('#leaderboard tbody tr:first-child button')
    if view_button:
        await view_button.click()
        await page.wait_for_timeout(2000)
        await page.screenshot(
            path=OUTPUT_DIR / f"08_experiments_model_detail_desktop{suffix}.png",
            full_page=True
        )
    
    # 9. Price Predictor Page
    print("  📷 Capturing: Price Predictor - Default View...")
    await page.click('button[data-page="predictor"]')
    await page.wait_for_timeout(3000)  # Wait for model to load and predict
    await page.screenshot(
        path=OUTPUT_DIR / f"09_predictor_default_desktop{suffix}.png",
        full_page=True
    )
    
    # 10. Price Predictor - Luxury Preset
    print("  📷 Capturing: Price Predictor - Luxury Preset...")
    await page.click('button:has-text("💎 Luxury")')
    await page.wait_for_timeout(1500)
    await page.screenshot(
        path=OUTPUT_DIR / f"10_predictor_luxury_desktop{suffix}.png",
        full_page=True
    )
    
    # 11. Price Predictor - 2021 Boom Market
    print("  📷 Capturing: Price Predictor - 2021 Boom...")
    await page.click('button:has-text("🚀 \'21 Boom")')
    await page.wait_for_timeout(1500)
    await page.screenshot(
        path=OUTPUT_DIR / f"11_predictor_2021_boom_desktop{suffix}.png",
        full_page=True
    )
    
    await context.close()


async def capture_mobile(browser, mode="light"):
    """Capture mobile screenshots in specified mode."""
    suffix = "_dark" if mode == "dark" else ""
    print(f"\n📱 Taking mobile screenshots ({mode} mode)...")
    
    context = await browser.new_context(
        viewport=VIEWPORT_MOBILE,
        device_scale_factor=3,  # High DPI for mobile
        is_mobile=True,
        has_touch=True,
        color_scheme=mode
    )
    page = await context.new_page()
    
    # Navigate to the dashboard
    await page.goto(BASE_URL, wait_until="networkidle")
    
    # Force the correct theme
    await set_theme(page, mode)
    
    await page.wait_for_timeout(2000)
    
    # 1. Mobile - Introduction
    print("  📷 Capturing: Mobile - Introduction...")
    await mobile_navigate(page, "intro", 1000)
    await page.screenshot(
        path=OUTPUT_DIR / f"mobile_01_intro{suffix}.png",
        full_page=False
    )
    
    # 2. Mobile - Data Story (Landscape)
    print("  📷 Capturing: Mobile - Data Story (Landscape)...")
    await mobile_navigate(page, "story", 2000)
    await page.screenshot(
        path=OUTPUT_DIR / f"mobile_02_story_landscape{suffix}.png",
        full_page=False
    )
    
    # 3. Mobile - Data Story (Ski Lift) - scroll to section
    print("  📷 Capturing: Mobile - Data Story (Ski Lift)...")
    await page.evaluate("""
        document.querySelector('[data-step="ski-lift"]').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    """)
    await page.wait_for_timeout(2000)
    await page.screenshot(
        path=OUTPUT_DIR / f"mobile_03_story_ski_lift{suffix}.png",
        full_page=False
    )
    
    # 4. Mobile - Data Story (Market Cycles)
    print("  📷 Capturing: Mobile - Data Story (Market Cycles)...")
    await page.evaluate("""
        document.querySelector('[data-step="market-cycles"]').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    """)
    await page.wait_for_timeout(2000)
    await page.screenshot(
        path=OUTPUT_DIR / f"mobile_04_story_market_cycles{suffix}.png",
        full_page=False
    )
    
    # 5. Mobile - Data Story (Buyer Origins)
    print("  📷 Capturing: Mobile - Data Story (Buyer Origins)...")
    await page.evaluate("""
        document.querySelector('[data-step="buyer-origins"]').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    """)
    await page.wait_for_timeout(2000)
    await page.screenshot(
        path=OUTPUT_DIR / f"mobile_05_story_buyer_origins{suffix}.png",
        full_page=False
    )
    
    # 6. Mobile - Data Story (Seasonality)
    print("  📷 Capturing: Mobile - Data Story (Seasonality)...")
    await page.evaluate("""
        document.querySelector('[data-step="seasonality"]').scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    """)
    await page.wait_for_timeout(2000)
    await page.screenshot(
        path=OUTPUT_DIR / f"mobile_06_story_seasonality{suffix}.png",
        full_page=False
    )
    
    # 7. Mobile - ML Experiments
    print("  📷 Capturing: Mobile - ML Experiments...")
    await mobile_navigate(page, "experiments", 2000)
    await page.screenshot(
        path=OUTPUT_DIR / f"mobile_07_experiments{suffix}.png",
        full_page=False
    )
    
    # 8. Mobile - Price Predictor (Default)
    print("  📷 Capturing: Mobile - Price Predictor (Default)...")
    await mobile_navigate(page, "predictor", 3000)
    await page.screenshot(
        path=OUTPUT_DIR / f"mobile_08_predictor_default{suffix}.png",
        full_page=False
    )
    
    # 9. Mobile - Price Predictor (Luxury Preset)
    print("  📷 Capturing: Mobile - Price Predictor (Luxury)...")
    await page.click('button:has-text("💎 Luxury")')
    await page.wait_for_timeout(1500)
    await page.screenshot(
        path=OUTPUT_DIR / f"mobile_09_predictor_luxury{suffix}.png",
        full_page=False
    )
    
    await context.close()


async def main():
    """Main entry point."""
    print("=" * 60)
    print("🏔️  Summit Housing Dashboard - Screenshot Tool")
    print("=" * 60)
    
    try:
        OUTPUT_DIR.mkdir(exist_ok=True)
        print(f"📸 Screenshots will be saved to: {OUTPUT_DIR}")

        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            
            # Light Mode (Default)
            await capture_desktop(browser, "light")
            await capture_mobile(browser, "light")
            
            # Dark Mode
            await capture_desktop(browser, "dark")
            await capture_mobile(browser, "dark")
            
            await browser.close()
            
        print(f"\n✅ All screenshots saved to: {OUTPUT_DIR}")
        print(f"📊 Total screenshots: {len(list(OUTPUT_DIR.glob('*.png')))}")
        print("\n✨ Screenshot capture complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
