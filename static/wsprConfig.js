// Enter your callsign and station coordinates below
const WSPR_tx_Callsign  = "HB9IIU";   // Your callsign
const WSPR_tx_Latitude  = 46.4668752; // Your station latitude
const WSPR_tx_Longitude = 6.8617024;  // Your station longitude

// To use the 3D Cesium globe, you need a free Cesium Ion API key.
// 1. Visit https://cesium.com/ion/
// 2. Sign up for a free account (or log in if you already have one).
// 3. Go to "Access Tokens" in your account settings.
// 4. Generate a new API token and copy it below.
//
// If you do not wish to use the 3D globe, leave it empty ("").
const cesiumAccessToken = "";

// Select which bands you want to display in the history (true = enabled, false = disabled)
const displayHistogram80meterBand = false;
const displayHistogram40meterBand = true;
const displayHistogram20meterBand = true;
const displayHistogram15meterBand = false;
const displayHistogram10meterBand = false;
