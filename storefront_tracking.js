/**
 * VIP Merchant Command Center - Storefront Tracking Script
 * Include this script in your E-commerce storefront to power the background Decision Intelligence Engine.
 * 
 * Embed: <script src="storefront_tracking.js"></script>
 */

(function () {
    const API_ENDPOINT = "http://localhost:8000/predict_live";
    const SESSION_KEY = "merchant_ai_session";

    let sessionData = JSON.parse(sessionStorage.getItem(SESSION_KEY) || getInitialSessionData());
    
    // Auto-update Start time
    const start_time = Date.now();

    function getInitialSessionData() {
        return JSON.stringify({
            visitor_id: "V-" + Math.floor(Math.random() * 9000 + 1000),
            Administrative: 0,
            Administrative_Duration: 0,
            Informational: 0,
            Informational_Duration: 0,
            ProductRelated: 1, // At least the first page
            ProductRelated_Duration: 0,
            BounceRates: 0.0,
            PageValues: 0.0,
            add_to_cart_count: 0,
            current_page: window.location.pathname
        });
    }

    function saveSession() {
        sessionStorage.setItem(SESSION_KEY, JSON.stringify(sessionData));
    }

    // Hook into navigations
    const originalPushState = history.pushState;
    history.pushState = function() {
        originalPushState.apply(this, arguments);
        sessionData.ProductRelated += 1;
        sessionData.current_page = window.location.pathname;
        saveSession();
    };

    // Passive Add To Cart click listener
    document.addEventListener("click", (e) => {
        const text = e.target.textContent?.toLowerCase() || "";
        if (text.includes("add to cart") || text.includes("buy")) {
            sessionData.add_to_cart_count += 1;
            sessionData.PageValues += 15.0; // Simulate an increase in page value
            saveSession();
        }
    }, { passive: true });

    // Sync loop every 12 seconds
    setInterval(() => {
        // Update durations
        sessionData.ProductRelated_Duration = (Date.now() - start_time) / 1000;
        
        // Push batch to our backend
        fetch(API_ENDPOINT, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                Administrative: sessionData.Administrative,
                Administrative_Duration: sessionData.Administrative_Duration,
                Informational: sessionData.Informational,
                Informational_Duration: sessionData.Informational_Duration,
                ProductRelated: sessionData.ProductRelated,
                ProductRelated_Duration: sessionData.ProductRelated_Duration,
                BounceRates: sessionData.BounceRates,
                PageValues: sessionData.PageValues
                // The API will auto-fill missing fields using smart defaults via Pydantic
            })
        }).catch(err => {
            // Silently fail if server isn't up
        });
        
    }, 12000);

})();
