document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('searchInput');
    const searchButton = document.getElementById('searchButton');
    const resultsContainer = document.getElementById('resultsContainer');
    const statusDiv = document.getElementById('status');
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValueSpan = document.getElementById('thresholdValue');


    const API_BASE = '/api';
    const CONTEXT_STEPS = 4; // Must match backend if used for indexing yellow_liner

    let lastResults = []; // Store the last fetched results

    // --- Helper function for Yellow Liner with Threshold ---
    function scoreToYellowBg(score, threshold) {
        // Ensure threshold is a number, default to 0 if invalid
        const validThreshold = (typeof threshold === 'number' && !isNaN(threshold)) ? threshold : 0;

        if (!score || score < validThreshold) {
            return '#FFFFFF'; // White below threshold
        }

        // Remap the score from [threshold, 1] to [0, 1] for color intensity calculation
        let remappedScore = 0;
        if (validThreshold < 1) { // Avoid division by zero if threshold is 1
            remappedScore = (score - validThreshold) / (1 - validThreshold);
        } else if (score >= 1) { // If threshold is 1, only scores >= 1 get full color
            remappedScore = 1;
        }
        remappedScore = Math.min(1, Math.max(0, remappedScore)); // Clamp 0-1

        // Decrease Blue channel based on remapped intensity
        const blueHex = Math.round(255 * (1 - remappedScore * 0.8)).toString(16).padStart(2, '0');
        return `#FFFF${blueHex}`;
    }

    // --- Helper to generate highlighted spans ---
    function generateHighlightedSpans(chunk, yellow_liner, threshold) {
        const spans = [];
        if (!yellow_liner || yellow_liner.length === 0) {
            // No highlighting data, return plain text node
            spans.push(document.createTextNode(chunk));
            return spans;
        }

        // Apply highlighting
        for (let i = 0; i < chunk.length; i++) {
            const char = chunk[i];
            const segmentIndex = Math.floor(i / CONTEXT_STEPS);
            // Use score 0 if index is out of bounds in yellow_liner
            const score = (segmentIndex < yellow_liner.length) ? yellow_liner[segmentIndex] : 0;
            const bgColor = scoreToYellowBg(score, threshold);

            if (char === '\n') {
                spans.push(document.createElement('br'));
            } else {
                const span = document.createElement('span');
                // Apply background only if above threshold (scoreToYellowBg returns white otherwise)
                if (bgColor !== '#FFFFFF') {
                    span.style.backgroundColor = bgColor;
                }
                span.textContent = char;
                spans.push(span);
            }
        }
        return spans;
    }


    // --- Function to render a single search result ---
    function renderResult(result, threshold) { // Accept threshold
        const itemDiv = document.createElement('div');
        itemDiv.className = 'result-item';

        const title = document.createElement('h4');
        const descParts = result.desc.split('/');
        const shortDesc = descParts[descParts.length - 1];
        title.textContent = `${shortDesc} (Score: ${result.cosine.toFixed(4)})`;
        itemDiv.appendChild(title);

        const chunkDiv = document.createElement('div');
        chunkDiv.className = 'result-chunk';

        // Generate spans using the helper and current threshold
        const spans = generateHighlightedSpans(result.chunk, result.yellow_liner, threshold);
        spans.forEach(span => chunkDiv.appendChild(span)); // Append generated spans/text nodes/brs

        itemDiv.appendChild(chunkDiv);
        return itemDiv;
    }

    // --- Function to redraw results with a new threshold ---
    function redrawResults(threshold) {
        resultsContainer.innerHTML = ''; // Clear previous rendering
        if (!lastResults || lastResults.length === 0) {
            // Display message if last search had no results (or no search yet)
            // resultsContainer.innerHTML = '<p>No results to display.</p>';
            return;
        }
        lastResults.forEach(result => {
            resultsContainer.appendChild(renderResult(result, threshold)); // Pass threshold
        });
    }

    // --- Function to perform the search ---
    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) {
            resultsContainer.innerHTML = '<p>Please enter a search query.</p>';
            lastResults = []; // Clear stored results
            return;
        }

        resultsContainer.innerHTML = '<p>Searching...</p>';
        lastResults = []; // Clear stored results before new search

        try {
            const response = await fetch(`${API_BASE}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    search_text: query,
                    max_results: 10,
                    yellow_liner: true // Always request yellow liner data
                }),
            });

            if (!response.ok) {
                let errorMsg = `Error: ${response.status} ${response.statusText}`;
                try { const errorData = await response.json(); errorMsg = errorData.error || errorData.detail || errorMsg; } catch (e) { }
                resultsContainer.innerHTML = `<p style="color: red;">${errorMsg}</p>`;
                return;
            }

            lastResults = await response.json(); // Store the new results

            // Perform initial render using the current slider threshold
            const currentThreshold = parseFloat(thresholdSlider.value);
            redrawResults(currentThreshold); // Call redraw function

        } catch (error) {
            console.error("Search fetch error:", error);
            resultsContainer.innerHTML = `<p style="color: red;">Network error or backend unavailable.</p>`;
            lastResults = []; // Clear results on error
        }
    }

    // --- Function to fetch and display status ---
    async function updateStatus() {
        // ... (status function remains the same) ...
        try {
            const response = await fetch(`${API_BASE}/status`);
            if (!response.ok) { statusDiv.textContent = 'Error fetching status.'; return; }
            const status = await response.json();
            statusDiv.textContent = `Status: ${status.library_size} docs | Model: ${status.current_model_name || 'None'} | Available: ${status.available_models.join(', ')}`;
        } catch (error) { console.error("Status fetch error:", error); statusDiv.textContent = 'Status unavailable.'; }
    }

    // --- Event Listeners ---
    searchButton.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') { performSearch(); }
    });

    // Slider Event Listener
    thresholdSlider.addEventListener('input', () => {
        const threshold = parseFloat(thresholdSlider.value);
        thresholdValueSpan.textContent = threshold.toFixed(2); // Update display
        redrawResults(threshold); // Redraw existing results with new threshold
    });

    // --- Initial Setup ---
    thresholdValueSpan.textContent = parseFloat(thresholdSlider.value).toFixed(2); // Set initial display
    updateStatus(); // Load initial status

});