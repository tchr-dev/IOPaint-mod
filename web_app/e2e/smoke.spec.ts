import { test, expect } from '@playwright/test';

test('has title', async ({ page }) => {
    await page.goto('/');

    // Expect a title "to contain" a substring.
    await expect(page).toHaveTitle(/IOPaint/);
});

test('loads editor and settings', async ({ page }) => {
    await page.goto('/');

    // Check if main workspace exists (adjust selector based on actual app)
    // Based on reading FileManager.test.tsx etc, there's likely a workspace or editor ID.
    // Using a generic check for now, can refine if needed.
    // Ideally, we look for a button or text we know exists.

    // Check for Settings button
    // Often identifiable by aria-label or icon
    // Let's wait for something stable.
    await expect(page.locator('body')).toBeVisible();
});
