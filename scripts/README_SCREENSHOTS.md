# 📸 Screenshot Tool for Portfolio

This script automatically captures high-quality screenshots of the Summit Housing Dashboard for use in your portfolio.

## Setup

1. **Install Playwright** (one-time setup):
   ```bash
   pip install playwright
   playwright install chromium
   ```

2. **Make sure your dashboard is running**:
   ```bash
   # In a separate terminal, start the local server
   cd static_dashboard
   python3 -m http.server 8000
   ```

## Usage

Run the screenshot script (this will capture both light and dark mode versions):
```bash
python3 scripts/take_screenshots.py
```

## Output

Screenshots will be saved to `screenshots/` directory with the following naming convention:

### Desktop Screenshots (1920x1080, 2x scale)
- `01_intro_desktop.png` - Introduction page
- `02_story_landscape_desktop.png` - Data Story: Landscape map
- `03_story_ski_lift_desktop.png` - Data Story: Ski lift proximity
- `04_story_market_cycles_desktop.png` - Data Story: Market trends
- `05_story_buyer_origins_desktop.png` - Data Story: Buyer origins
- `06_story_seasonality_desktop.png` - Data Story: Seasonal patterns
- `07_experiments_leaderboard_desktop.png` - ML Experiments leaderboard
- `08_experiments_model_detail_desktop.png` - ML Experiments model detail
- `09_predictor_default_desktop.png` - Price Predictor default view
- `10_predictor_luxury_desktop.png` - Price Predictor luxury preset
- `11_predictor_2021_boom_desktop.png` - Price Predictor 2021 boom scenario

### Mobile Screenshots (430x932, 3x scale - iPhone 14 Pro Max)
- `mobile_01_intro.png` - Mobile introduction
- `mobile_02_story_landscape.png` - Mobile data story: Landscape
- `mobile_03_story_ski_lift.png` - Mobile data story: Ski lift
- `mobile_04_story_market_cycles.png` - Mobile data story: Market cycles
- `mobile_05_story_buyer_origins.png` - Mobile data story: Buyer origins
- `mobile_06_story_seasonality.png` - Mobile data story: Seasonality
- `mobile_07_experiments.png` - Mobile ML experiments
- `mobile_08_predictor_default.png` - Mobile price predictor: Default
- `mobile_09_predictor_luxury.png` - Mobile price predictor: Luxury

## Features

- ✅ Retina-quality screenshots (2x scale for desktop, 3x for mobile)
- ✅ Captures all major sections of the dashboard
- ✅ Includes both desktop and mobile views
- ✅ Automatically waits for content to load
- ✅ Full-page screenshots where appropriate
- ✅ Numbered files for easy organization

### Dark Mode
Dark mode screenshots follow the same naming convention but with a `_dark` suffix:
- `01_intro_desktop_dark.png`
- `mobile_01_intro_dark.png`
- etc.

## Customization

Edit `scripts/take_screenshots.py` to:
- Change viewport sizes (`VIEWPORT_DESKTOP`, `VIEWPORT_MOBILE`)
- Adjust wait times for slower connections
- Add or remove specific sections
- Change output directory
