import * as THREE from './three.module.js';
import { OrbitControls } from './OrbitControls.js';

// Global variables
let scene, camera, renderer, controls;
let pointsObject = null; // THREE.Points object
let connectionLines = [];
let raycaster, mouse;
let selectedPointIndex = null; // Index of the *clicked point*
let originalColors = []; // Stores original [r,g,b] float components
let statsElement;
let docData = null; // Will hold the columnar JSON data
let currentDataUrl = '';
let infoDiv = null; // Declare globally, assign in init
let animationFrameId = null;

function onWindowResize() {
    if (camera && renderer) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

function resetView() {
    if (camera && controls) {
        camera.position.set(0, 0, 5);
        controls.target.set(0, 0, 0);
        controls.reset();
        if (selectedPointIndex !== null) { // Clear highlight
            highlightPoint(selectedPointIndex, true); // Pass true to force deselect
        }
        if (infoDiv) infoDiv.innerHTML = 'Click a point to see document info';
    }
}

function highlightPoint(pointIndexOfClicked, forceDeselect = false) {
    if (!pointsObject || !pointsObject.geometry || !docData || !docData.doc_ids || !docData.colors) {
        // console.warn("highlightPoint: Prerequisites not met.");
        return;
    }

    const colorsAttribute = pointsObject.geometry.attributes.color;
    if (!colorsAttribute) {
        // console.warn("highlightPoint: colorsAttribute is null");
        return;
    }
    if (!originalColors || originalColors.length !== colorsAttribute.count * 3) {
        // console.warn("highlightPoint: originalColors is not populated correctly or length mismatch.");
        return;
    }

    // Determine if this action will result in a deselection
    const isDeselecting = forceDeselect || (selectedPointIndex !== null && selectedPointIndex === pointIndexOfClicked);

    if (isDeselecting) {
        // DESELECTION: Restore ALL points to their original full brightness
        for (let i = 0; i < colorsAttribute.count; i++) {
            colorsAttribute.setXYZ(
                i,
                originalColors[i * 3],       // Full original R
                originalColors[i * 3 + 1],   // Full original G
                originalColors[i * 3 + 2]    // Full original B
            );
        }
        selectedPointIndex = null; // Clear the global selected point index
        // console.log("Deselecting. All points restored to original brightness.");
    } else {
        // SELECTION or CHANGING SELECTION:
        // This means a new point (potentially from a new document) is being selected.

        // 1. Dim ALL points first
        for (let i = 0; i < colorsAttribute.count; i++) {
            colorsAttribute.setXYZ(
                i,
                originalColors[i * 3] * 0.3,     // Dim R
                originalColors[i * 3 + 1] * 0.3, // Dim G
                originalColors[i * 3 + 2] * 0.3  // Dim B
            );
        }

        // 2. Then, brighten points of the newly selected document
        selectedPointIndex = pointIndexOfClicked; // Update the global selected point index
        const clickedDocId = docData.doc_ids[pointIndexOfClicked];
        // console.log(`Selecting document: ${clickedDocId}`);

        for (let i = 0; i < docData.doc_ids.length; i++) {
            if (docData.doc_ids[i] === clickedDocId) {
                colorsAttribute.setXYZ(
                    i,
                    originalColors[i * 3],       // Restore full R for selected doc's points
                    originalColors[i * 3 + 1],   // Restore full G
                    originalColors[i * 3 + 2]    // Restore full B
                );
            }
        }
    }
    colorsAttribute.needsUpdate = true;
}

function onMouseClick(event) {
    if (!docData || !pointsObject || !camera || !raycaster || !mouse) return;

    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    const intersects = raycaster.intersectObject(pointsObject);

    if (intersects.length > 0) {
        const pointIndex = intersects[0].index;

        if (pointIndex >= docData.points.length) {
            console.error("Clicked point index out of bounds for docData arrays.");
            return;
        }

        // Use doc_ids here
        const docIdForClickedPoint = docData.doc_ids[pointIndex];
        const textForClickedPoint = docData.texts[pointIndex];

        if (infoDiv) {
            infoDiv.innerHTML = `
                <strong>Document:</strong> ${docIdForClickedPoint || "N/A"}<br>
                <strong>Text:</strong> ${textForClickedPoint ? textForClickedPoint.substring(0, 200) + "..." : "N/A"}
            `;
        }
        highlightPoint(pointIndex);
    } else {
        if (selectedPointIndex !== null) {
            highlightPoint(selectedPointIndex, true);
        }
        if (infoDiv) infoDiv.innerHTML = 'Click a point to see document info';
    }
}

function updatePointSize() {
    const pointSizeSlider = document.getElementById('pointSize');
    if (pointSizeSlider && pointsObject && pointsObject.material) { // Check pointsObject.material
        const newSize = parseFloat(pointSizeSlider.value);
        pointsObject.material.size = newSize;
        if (raycaster && raycaster.params.Points) {
            raycaster.params.Points.threshold = newSize * 0.1; // Adjust threshold with size
        }
    } else {
        if (!pointSizeSlider) console.warn("Element with ID 'pointSize' not found.");
        if (!pointsObject || !pointsObject.material) console.warn("'pointsObject.material' is not defined or not yet initialized.");
    }
}

function toggleConnections(event) {
    const visible = event.target.checked;
    connectionLines.forEach(line => {
        line.visible = visible;
    });
}

export function initVisualization(dataUrl) {
    currentDataUrl = dataUrl;

    // Initialize infoDiv here after DOM is ready
    if (!infoDiv) { // Ensure it's only fetched once or if it was null
        infoDiv = document.getElementById('info');
        if (!infoDiv) {
            console.error("HTML element with ID 'info' not found! Click information will not be displayed.");
        }
    }


    if (scene) { // Clear previous scene objects
        while (scene.children.length > 0) {
            const obj = scene.children[0];
            scene.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) {
                    obj.material.forEach(material => material.dispose());
                } else {
                    obj.material.dispose();
                }
            }
        }
        connectionLines.forEach(line => { // Also dispose connection line geometries/materials
            if (line.geometry) line.geometry.dispose();
            if (line.material) line.material.dispose();
        });
        connectionLines = [];
        pointsObject = null;
        selectedPointIndex = null;
        originalColors = [];
        // docData = null; // Keep docData if you want to compare, or clear if fresh load always
    }


    if (!renderer) { // One-time setup
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        scene = new THREE.Scene();

        camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 5;

        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        raycaster = new THREE.Raycaster();
        raycaster.params.Points.threshold = 0.05; // Initial, will be updated by updatePointSize
        mouse = new THREE.Vector2();

        statsElement = document.getElementById('stats');

        document.getElementById('resetView').addEventListener('click', resetView);
        document.getElementById('pointSize').addEventListener('input', updatePointSize); // updatePointSize will read value
        document.getElementById('showConnections').addEventListener('change', toggleConnections);

        window.addEventListener('click', onMouseClick);
        window.addEventListener('resize', onWindowResize);

        animate();
    }

    // Always set background and add lights as they might be removed if scene was cleared
    scene.background = new THREE.Color(0xf0f0f0);
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1).normalize();
    scene.add(directionalLight);

    if (infoDiv) infoDiv.innerHTML = 'Loading data...';
    loadData(currentDataUrl);
}

// IMPORTANT: You will need to include oboe.js in your HTML file for this to work.
// e.g., <script src="https://cdnjs.cloudflare.com/ajax/libs/oboe.js/2.1.5/oboe-browser.min.js"></script>

function loadData(dataUrl) {
    const loadingElement = document.getElementById('loading');
    const progressBar = document.querySelector('.progress-bar');
    if (loadingElement) loadingElement.style.display = 'block';
    if (progressBar) progressBar.style.width = '0%';
    const loadingTextDiv = loadingElement ? loadingElement.querySelector('div') : null;
    if (loadingTextDiv) loadingTextDiv.textContent = 'Loading visualization data...';

    // This promise will resolve with the parsed data or reject with an error
    const dataProcessingPromise = new Promise((resolve, reject) => {
        fetch(dataUrl)
            .then(response => {
                const outerResponse = response;
                if (!outerResponse.ok) {
                    reject(new Error(`HTTP error! status: ${outerResponse.status}`));
                    return;
                }

                const contentLength = outerResponse.headers.get('Content-Length');
                const totalLength = contentLength ? parseInt(contentLength) : null;

                if (!outerResponse.body) {
                    reject(new Error('Response body is null.'));
                    return;
                }

                // Use Oboe.js for large files (e.g., > 10MB), standard .json() for smaller files
                if (totalLength && totalLength > 10000000) {
                    if (loadingTextDiv) loadingTextDiv.textContent = 'Loading large dataset (Oboe.js streaming parse)...';

                    let oboeInstance = oboe(); // Create an Oboe instance
                    oboeInstance
                        .done(data => { // Called when Oboe successfully parses the entire JSON
                            if (loadingTextDiv) loadingTextDiv.textContent = 'JSON parsing complete. Processing visualization...';
                            console.log("Oboe.js successfully parsed the root JSON object.");
                            docData = data; // Store the parsed data
                            resolve(data);  // Resolve the main promise with the data
                        })
                        .fail(errorReport => { // Called if Oboe encounters a parsing error
                            console.error("Oboe.js parsing error:", errorReport);
                            if (loadingTextDiv) loadingTextDiv.textContent = 'Error parsing JSON stream.';
                            const thrownError = errorReport.thrown || {};
                            reject(new Error(`Oboe.js parsing failed: ${thrownError.message || 'Unknown parsing error'}. Status: ${errorReport.statusCode}, Body: ${errorReport.body}`));
                        });

                    const reader = outerResponse.body.getReader();
                    const decoder = new TextDecoder("utf-8", { stream: true }); // Use streaming TextDecoder
                    let receivedLength = 0;

                    function processStreamedChunks({ done, value }) {
                        if (oboeInstance.failed) { // If Oboe already failed, stop processing
                            console.warn("Oboe instance reported failure, cancelling stream read.");
                            reader.cancel().catch(cancelError => console.error("Error cancelling reader:", cancelError));
                            return;
                        }

                        if (done) {
                            const lastChunkString = decoder.decode(); // Flush any remaining bytes in the decoder
                            if (lastChunkString) {
                                try {
                                    oboeInstance.emit('data', lastChunkString);
                                } catch (e) {
                                    // This error should ideally be caught by Oboe's .fail()
                                    console.error("Error emitting final data chunk to Oboe:", e);
                                    if (!oboeInstance.failed) reject(e); // Reject if Oboe hasn't already
                                    return;
                                }
                            }
                            try {
                                oboeInstance.emit('end'); // Signal end of data stream to Oboe
                                console.log("Emitted 'end' to Oboe.js");
                            } catch (e) {
                                console.error("Error emitting 'end' to Oboe:", e);
                                if (!oboeInstance.failed) reject(e);
                            }

                            if (totalLength && receivedLength !== totalLength) {
                                console.warn(`Stream ended but receivedLength (${receivedLength}) does not match Content-Length (${totalLength}).`);
                            }
                            // Oboe's 'done' or 'fail' handlers will resolve/reject the main promise.
                            return;
                        }

                        // Decode the current chunk and emit it to Oboe
                        const chunkString = decoder.decode(value, { stream: true });
                        if (chunkString) {
                            try {
                                oboeInstance.emit('data', chunkString);
                            } catch (e) {
                                console.error("Error emitting data chunk to Oboe:", e, "Chunk string length:", chunkString.length);
                                if (!oboeInstance.failed) reject(e);
                                reader.cancel().catch(cancelError => console.error("Error cancelling reader:", cancelError));
                                return;
                            }
                        }

                        receivedLength += value.length;

                        // Update progress bar
                        if (progressBar && totalLength) {
                            const percentage = Math.round((receivedLength / totalLength) * 100);
                            progressBar.style.width = percentage + '%';
                            if (loadingTextDiv) loadingTextDiv.textContent = `Loading large dataset... ${percentage}% (${Math.round(receivedLength / 1024 / 1024)}MB / ${Math.round(totalLength / 1024 / 1024)}MB)`;
                        } else if (loadingTextDiv) {
                            loadingTextDiv.textContent = `Loading large dataset... ${Math.round(receivedLength / 1024 / 1024)}MB received`;
                        }

                        // Continue reading the stream
                        reader.read().then(processStreamedChunks).catch(streamError => {
                            console.error("Error reading stream:", streamError);
                            if (!oboeInstance.failed) oboeInstance.emit('error', streamError); // Inform Oboe
                            reject(streamError); // Reject the main promise
                        });
                    }
                    // Start reading the stream
                    reader.read().then(processStreamedChunks).catch(initialReadError => {
                        console.error("Error on initial stream read:", initialReadError);
                        if (!oboeInstance.failed) oboeInstance.emit('error', initialReadError);
                        reject(initialReadError);
                    });

                } else {
                    // For smaller files (or if Content-Length is unknown but assumed small)
                    if (loadingTextDiv) loadingTextDiv.textContent = 'Loading dataset (standard JSON parse)...';
                    outerResponse.json()
                        .then(data => {
                            docData = data;
                            resolve(data); // Resolve the main promise
                        })
                        .catch(jsonParseError => {
                            console.error("Standard JSON.parse error:", jsonParseError);
                            reject(jsonParseError); // Reject the main promise
                        });
                }
            })
            .catch(fetchError => { // Catch errors from the fetch() call itself
                console.error("Fetch error:", fetchError);
                reject(fetchError); // Reject the main promise
            });
    });

    // Handle the result of the data processing promise
    dataProcessingPromise
        .then(parsedData => {
            // This block executes if data loading and parsing (either by Oboe or response.json()) was successful
            if (loadingTextDiv) loadingTextDiv.textContent = 'Processing visualization...';
            else if (loadingElement && loadingElement.querySelector('div')) {
                loadingElement.querySelector('div').textContent = 'Processing visualization...';
            }

            setTimeout(() => { // Process visualization on the next tick to allow UI updates
                try {
                    createVisualization(parsedData);
                    if (loadingElement) loadingElement.style.display = 'none';
                    updateStats();
                } catch (error) {
                    if (loadingElement) loadingElement.innerHTML = `<div class="error">Error creating visualization: ${error.message}</div>`;
                    console.error('Visualization error:', error);
                    if (infoDiv) infoDiv.innerHTML = `<span style="color:red;">Error during visualization. See console.</span>`;
                }
            }, 10);
        })
        .catch(error => {
            // This block executes if any error occurred during fetch, streaming, or parsing
            console.error('Overall error in loadData process:', error);
            if (loadingElement) loadingElement.innerHTML = `<div class="error">Failed to load or process data: ${error.message}</div>`;
            if (infoDiv) infoDiv.innerHTML = `<span style="color:red;">Failed to load data. Check console for details.</span>`;
        });
}

function createVisualization(data) { // data is docData
    if (!data || !data.points || !data.colors || !data.doc_ids || !data.texts) {
        console.error("createVisualization: Missing critical data fields (points, colors, doc_ids, texts).", data);
        if (infoDiv) infoDiv.innerHTML = "<span style='color:red;'>Error: Incomplete data for visualization.</span>";
        return;
    }

    const positionsArrays = data.points;
    const colorsArraysFromJSON = data.colors; // Array of [R,G,B] as 0-255 integers

    // --- ADDED LOGS FOR COLORS (can be kept or removed after confirming fix) ---
    if (colorsArraysFromJSON && colorsArraysFromJSON.length > 0) {
        console.log("Raw colorsArrays from JSON (first 5):", JSON.stringify(colorsArraysFromJSON.slice(0, 5)));
    } else {
        console.error("colorsArraysFromJSON is missing or empty in the JSON data!");
    }
    // --- END ADDED LOGS ---

    if (positionsArrays.length === 0) {
        console.warn("No points to visualize.");
        if (statsElement) statsElement.textContent = "No data points.";
        if (infoDiv) infoDiv.textContent = "No data points to display.";
        return;
    }
    if (positionsArrays.length !== colorsArraysFromJSON.length || positionsArrays.length !== data.doc_ids.length || positionsArrays.length !== data.texts.length) {
        console.error("Data array length mismatch!", /* ... */);
        if (infoDiv) infoDiv.innerHTML = "<span style='color:red;'>Error: Data inconsistency (array lengths differ).</span>";
        return;
    }

    // ... (normalization logic for positionsArrays remains the same) ...
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const pos of positionsArrays) {
        minX = Math.min(minX, pos[0]); maxX = Math.max(maxX, pos[0]);
        minY = Math.min(minY, pos[1]); maxY = Math.max(maxY, pos[1]);
        minZ = Math.min(minZ, pos[2]); maxZ = Math.max(maxZ, pos[2]);
    }
    const rangeX = maxX - minX || 1; const rangeY = maxY - minY || 1; const rangeZ = maxZ - minZ || 1;
    const maxRange = Math.max(rangeX, rangeY, rangeZ) || 1;

    const normalizedPositionsFlat = [];
    for (const pos of positionsArrays) {
        normalizedPositionsFlat.push(((pos[0] - minX) / maxRange * 10) - 5);
        normalizedPositionsFlat.push(((pos[1] - minY) / maxRange * 10) - 5);
        normalizedPositionsFlat.push(((pos[2] - minZ) / maxRange * 10) - 5);
    }

    // --- NORMALIZE COLORS from 0-255 integers to 0.0-1.0 floats ---
    const normalizedColorsFlat = [];
    for (const rgbArray of colorsArraysFromJSON) {
        normalizedColorsFlat.push(rgbArray[0] / 255.0); // Normalize R
        normalizedColorsFlat.push(rgbArray[1] / 255.0); // Normalize G
        normalizedColorsFlat.push(rgbArray[2] / 255.0); // Normalize B
    }
    const flatColorsForBuffer = new Float32Array(normalizedColorsFlat);
    // --- END COLOR NORMALIZATION ---


    // --- ADDED LOGS FOR NORMALIZED COLORS (can be kept or removed) ---
    if (flatColorsForBuffer && flatColorsForBuffer.length > 0) {
        console.log("Normalized and flattened color components for buffer (first 15):", flatColorsForBuffer.slice(0, 15));
    } else {
        console.error("flatColorsForBuffer array is empty after processing colorsArraysFromJSON!");
    }
    // --- END ADDED LOGS ---
    originalColors = [...flatColorsForBuffer]; // Store the normalized 0.0-1.0 colors

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(normalizedPositionsFlat, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(flatColorsForBuffer, 3)); // Use normalized colors
    geometry.computeBoundingSphere();

    const material = new THREE.PointsMaterial({
        size: parseFloat(document.getElementById('pointSize').value) || 0.05,
        vertexColors: true,
        sizeAttenuation: true,
    });

    pointsObject = new THREE.Points(geometry, material);
    scene.add(pointsObject);
    updatePointSize();

    if (infoDiv) infoDiv.textContent = 'Click a point to see document info';
    updateStats(); // Call updateStats after data is processed
}

function animate() {
    animationFrameId = requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}

function updateStats() {
    // Ensure statsElement is assigned, typically in initVisualization or setupThreeJS
    if (!statsElement) {
        statsElement = document.getElementById('stats'); // Attempt to get it if not already set
        if (!statsElement) {
            console.warn("updateStats: HTML element with ID 'stats' not found in DOM.");
            return;
        }
    }

    if (docData && docData.points) { // Basic check: do we have points data?
        let docCount = 0;
        let modelName = docData.model_name || "N/A"; // Get model name, default if not present

        // Prioritize doc_map for document count
        if (docData.doc_map && typeof docData.doc_map === 'object' && Object.keys(docData.doc_map).length > 0) {
            docCount = Object.keys(docData.doc_map).length;
            // console.log("updateStats: Using doc_map for document count:", docCount);
        }
        // Fallback to counting unique doc_ids if doc_map is not suitable or not present
        else if (docData.doc_ids && Array.isArray(docData.doc_ids) && docData.doc_ids.length > 0) {
            docCount = new Set(docData.doc_ids).size;
            // console.log("updateStats: Using unique doc_ids for document count (fallback):", docCount);
        } else {
            // console.log("updateStats: Could not determine document count from doc_map or doc_ids.");
        }

        statsElement.textContent = `Points: ${docData.points.length} | Documents: ${docCount} | Model: ${modelName}`;

    } else {
        statsElement.textContent = 'Statistics not available.';
        // Extended logging for when stats are not available
        console.warn("updateStats: Conditions not met to display stats. Current state of docData:");
        if (!docData) {
            console.warn("docData is null or undefined.");
        } else {
            console.warn(`docData.points: ${docData.points ? docData.points.length + ' points' : 'missing or empty'}`);
            console.warn(`docData.model_name: ${docData.model_name || 'missing'}`);
            console.warn(`docData.doc_map: ${docData.doc_map ? 'exists (' + Object.keys(docData.doc_map).length + ' keys)' : 'missing or not an object'}`);
            if (docData.doc_map && typeof docData.doc_map !== 'object') {
                console.warn(`docData.doc_map is of type: ${typeof docData.doc_map}`);
            }
            console.warn(`docData.doc_ids: ${docData.doc_ids ? (Array.isArray(docData.doc_ids) ? docData.doc_ids.length + ' entries' : 'not an array') : 'missing'}`);
        }
    }
}
// Ensure this script doesn't run init() by itself anymore, it will be called from HTML.
// init(); // Remove this line