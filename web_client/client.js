import * as THREE from './third_party/three/three.module.js';
import { OrbitControls } from './third_party/three/OrbitControls.js';

// --- Visualization Globals ---
let scene, camera, renderer, controls;
let pointsObject = null;
let raycaster, mouse;
let selectedPointIndex = null;
let originalColors = [];
let docData = null;
let animationFrameId = null;
let vizContainer = null;
let infoDiv = null;
let sendRequest = null;

// --- Visualization Functions ---

function initVisualization(container) {
    vizContainer = container;
    vizContainer.style.position = 'relative'; // Ensure relative positioning for absolute children if any

    // Info overlay
    infoDiv = document.createElement('div');
    infoDiv.style.position = 'absolute';
    infoDiv.style.top = '10px';
    infoDiv.style.left = '10px';
    infoDiv.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
    infoDiv.style.padding = '5px';
    infoDiv.style.borderRadius = '3px';
    infoDiv.style.pointerEvents = 'none'; // Let clicks pass through
    infoDiv.style.fontSize = '12px';
    infoDiv.style.color = '#000';
    infoDiv.innerText = 'Select a model to load visualization.';
    vizContainer.appendChild(infoDiv);

    const width = vizContainer.clientWidth;
    const height = vizContainer.clientHeight;

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    vizContainer.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.z = 5;

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.05;
    mouse = new THREE.Vector2();

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1).normalize();
    scene.add(directionalLight);

    // Events
    window.addEventListener('resize', onWindowResize);
    renderer.domElement.addEventListener('click', onMouseClick);

    animate();
}

function onWindowResize() {
    if (camera && renderer && vizContainer) {
        const width = vizContainer.clientWidth;
        const height = vizContainer.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    }
}

function onMouseClick(event) {
    if (!docData || !pointsObject || !camera || !raycaster || !mouse) return;

    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    const intersects = raycaster.intersectObject(pointsObject);

    if (intersects.length > 0) {
        const pointIndex = intersects[0].index;
        highlightPoint(pointIndex);

        if (docData.hashes && docData.chunk_indices) {
            const hash = docData.hashes[pointIndex];
            const chunkIndex = docData.chunk_indices[pointIndex];

            if (infoDiv) infoDiv.innerText = `Loading details...`;

            // Fetch details
            if (sendRequest) sendRequest('get_chunk_details', { hash: hash, chunk_index: chunkIndex });
        }
    } else {
        if (selectedPointIndex !== null) {
            highlightPoint(selectedPointIndex, true); // Deselect
        }
        infoDiv.innerText = 'Click a point to see info.';
    }
}

function highlightPoint(pointIndex, forceDeselect = false) {
    if (!pointsObject || !pointsObject.geometry) return;

    const colorsAttribute = pointsObject.geometry.attributes.color;
    const isDeselecting = forceDeselect || (selectedPointIndex === pointIndex);

    if (isDeselecting) {
        // Restore all
        for (let i = 0; i < colorsAttribute.count; i++) {
            colorsAttribute.setXYZ(i, originalColors[i * 3], originalColors[i * 3 + 1], originalColors[i * 3 + 2]);
        }
        selectedPointIndex = null;
    } else {
        // Dim all
        for (let i = 0; i < colorsAttribute.count; i++) {
            colorsAttribute.setXYZ(i, originalColors[i * 3] * 0.1, originalColors[i * 3 + 1] * 0.1, originalColors[i * 3 + 2] * 0.1);
        }
        // Highlight selected
        selectedPointIndex = pointIndex;

        // Also highlight other points from the same document if possible
        let pointsToHighlight = [pointIndex];
        if (docData.hashes) {
            const hash = docData.hashes[pointIndex];
            for (let i = 0; i < docData.hashes.length; i++) {
                if (docData.hashes[i] === hash) {
                    pointsToHighlight.push(i);
                }
            }
        }

        for (const idx of pointsToHighlight) {
            colorsAttribute.setXYZ(idx, originalColors[idx * 3], originalColors[idx * 3 + 1], originalColors[idx * 3 + 2]);
            // Maybe make it even brighter or a specific color?
            // For now just original color against dimmed background is good.
        }
    }
    colorsAttribute.needsUpdate = true;
}

function highlightDescriptors(descriptors) {
    if (!pointsObject || !pointsObject.geometry || !docData) return;

    const colorsAttribute = pointsObject.geometry.attributes.color;

    // Dim all first
    for (let i = 0; i < colorsAttribute.count; i++) {
        colorsAttribute.setXYZ(i, 0.8, 0.8, 0.8); // Make everything gray/dim
    }

    // Highlight matches
    const descriptorSet = new Set(descriptors);
    let matchCount = 0;

    if (docData.hashes) {
        for (let i = 0; i < docData.hashes.length; i++) {
            if (descriptorSet.has(docData.hashes[i])) {
                colorsAttribute.setXYZ(i, originalColors[i * 3], originalColors[i * 3 + 1], originalColors[i * 3 + 2]);
                matchCount++;
            }
        }
    }

    colorsAttribute.needsUpdate = true;
    console.log(`Highlighted ${matchCount} points for ${descriptors.length} descriptors.`);
}


function loadVisualizationData(modelName, sendCmd) {
    if (infoDiv) infoDiv.innerText = `Loading data for ${modelName}...`;

    // We use the websocket command to get data? Or fetch?
    // The plan said "Expose 3D data via API". I implemented 'get_3d_viz_data' command in websocket.
    // So we should send a command.

    sendCmd('get_3d_viz_data', modelName);
}

function createSceneFromData(data) {
    if (scene) {
        // Remove old points
        if (pointsObject) {
            scene.remove(pointsObject);
            pointsObject.geometry.dispose();
            pointsObject.material.dispose();
            pointsObject = null;
        }
    }

    docData = data;
    if (!docData || !docData.points) {
        if (infoDiv) infoDiv.innerText = 'No data available.';
        return;
    }

    const positions = [];
    const colors = [];

    // Normalize positions
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const pos of docData.points) {
        minX = Math.min(minX, pos[0]); maxX = Math.max(maxX, pos[0]);
        minY = Math.min(minY, pos[1]); maxY = Math.max(maxY, pos[1]);
        minZ = Math.min(minZ, pos[2]); maxZ = Math.max(maxZ, pos[2]);
    }
    const range = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;

    for (const pos of docData.points) {
        positions.push(((pos[0] - minX) / range * 10) - 5);
        positions.push(((pos[1] - minY) / range * 10) - 5);
        positions.push(((pos[2] - minZ) / range * 10) - 5);
    }

    // Process colors from hashes
    if (docData.hashes) {
        const colorMap = {};
        const maxChunkMap = {};
        docData.pointMap = {}; // Initialize point map

        // First pass: find max chunk index for each hash
        if (docData.chunk_indices) {
            for (let i = 0; i < docData.hashes.length; i++) {
                const hash = docData.hashes[i];
                const chunkIndex = docData.chunk_indices[i];
                if (!maxChunkMap[hash] || chunkIndex > maxChunkMap[hash]) {
                    maxChunkMap[hash] = chunkIndex;
                }
            }
        }

        for (let i = 0; i < docData.hashes.length; i++) {
            const hash = docData.hashes[i];

            // Build point map
            if (docData.chunk_indices) {
                const chunkIndex = docData.chunk_indices[i];
                const key = `${hash}:${chunkIndex}`;
                docData.pointMap[key] = i;
            }

            if (!colorMap[hash]) {
                // Generate base color from hash
                let h = 0;
                for (let j = 0; j < hash.length; j++) {
                    h = hash.charCodeAt(j) + ((h << 5) - h);
                }
                const hue = Math.abs(h % 360) / 360;
                colorMap[hash] = hue;
            }

            const hue = colorMap[hash];
            const color = new THREE.Color();

            // Apply gradient based on chunk index
            let lightness = 0.5;
            if (docData.chunk_indices && maxChunkMap[hash] > 0) {
                const chunkIndex = docData.chunk_indices[i];
                const maxChunk = maxChunkMap[hash];
                // Gradient from 0.6 (light) to 0.3 (dark)
                lightness = 0.6 - (chunkIndex / maxChunk) * 0.3;
            }

            color.setHSL(hue, 0.6, lightness);
            colors.push(color.r, color.g, color.b);
        }
    } else {
        // Default color if missing
        for (let i = 0; i < docData.points.length; i++) {
            colors.push(0, 0.5, 1);
        }
    }

    originalColors = [...colors];

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({ size: 0.05, vertexColors: true, sizeAttenuation: true });
    pointsObject = new THREE.Points(geometry, material);
    scene.add(pointsObject);

    if (infoDiv) infoDiv.innerText = `Loaded ${docData.points.length} points.`;
}

function animate() {
    animationFrameId = requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}


// --- Main Client Logic ---

window.onload = function () {
    console.log("Bootloader started");

    const themes = {
        light: {
            base03: '#002b36',
            base02: '#073642',
            base01: '#586e75',
            base00: '#657b83',
            base0: '#839496',
            base1: '#93a1a1',
            base2: '#eee8d5',
            base3: '#fdf6e3',
            yellow: '#b58900',
            orange: '#cb4b16',
            red: '#dc322f',
            magenta: '#d33682',
            violet: '#6c71c4',
            blue: '#268bd2',
            cyan: '#2aa198',
            green: '#859900',

            background: '#fdf6e3',
            foreground: '#657b83',
            primary: '#268bd2',
            secondary: '#93a1a1',
            border: '#93a1a1',
            highlight: '#eee8d5',
            error: '#dc322f',
            success: '#859900',
            warning: '#b58900',

            statusBarBg: '#268bd2',
            statusBarFg: '#fdf6e3',
            inputBg: 'transparent',
            inputFg: '#657b83',
            selectionBg: '#eee8d5',

            searchId: '#2aa198',
            searchScore: '#cb4b16',
            searchTitle: '#657b83',
            searchMeta: '#93a1a1',
            searchDesc: '#586e75',

            logLog: '#268bd2',
            logResponse: '#cb4b16',
            logUnknown: '#b58900',
            logError: '#dc322f',

            progressContainer: '#eee8d5',
            progressBar: '#268bd240',
            progressLabel: '#657b83',
        }
    };

    const theme = themes.light;

    // --- Layout ---
    const style = document.createElement('style');
    style.innerHTML = `
        * {
            box-sizing: border-box;
        }
        /* Scrollbar styling for WebKit (Chrome, Safari, Edge) */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: ${theme.base2}; 
        }
        ::-webkit-scrollbar-thumb {
            background: ${theme.base00}; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: ${theme.base01}; 
        }
    `;
    document.head.appendChild(style);

    document.body.style.margin = '0';
    document.body.style.height = '100vh';
    document.body.style.display = 'flex';
    document.body.style.flexDirection = 'column';
    document.body.style.fontFamily = 'sans-serif';
    document.body.style.backgroundColor = theme.background;
    document.body.style.color = theme.foreground;

    // Status Bar
    const statusBar = document.createElement('div');
    statusBar.style.height = '25px';
    statusBar.style.backgroundColor = theme.statusBarBg;
    statusBar.style.color = theme.statusBarFg;
    statusBar.style.display = 'flex';
    statusBar.style.alignItems = 'center';
    statusBar.style.padding = '0 10px';
    statusBar.style.fontSize = '12px';
    statusBar.innerText = 'Connecting...';
    document.body.appendChild(statusBar);

    // Main Container (3 columns)
    const mainContainer = document.createElement('div');
    mainContainer.style.flex = '1';
    mainContainer.style.display = 'flex';
    mainContainer.style.overflow = 'hidden';
    mainContainer.style.borderBottom = `1px solid ${theme.border}`;
    document.body.appendChild(mainContainer);

    // Column 1: Property (20%)
    const colProperty = document.createElement('div');
    colProperty.style.flex = '0 0 20%';
    colProperty.style.borderRight = `1px solid ${theme.border}`;
    colProperty.style.overflowY = 'auto';
    colProperty.style.padding = '10px';
    mainContainer.appendChild(colProperty);

    // Column 2: Main (40%)
    const colMain = document.createElement('div');
    colMain.style.flex = '0 0 40%';
    colMain.style.borderRight = `1px solid ${theme.border}`;
    colMain.style.overflowY = 'auto';
    colMain.style.padding = '10px';
    colMain.innerText = 'Main Content Area';
    colMain.style.backgroundColor = theme.background;
    mainContainer.appendChild(colMain);

    // Column 3: Detail (40%)
    const colDetail = document.createElement('div');
    colDetail.style.flex = '0 0 40%';
    colDetail.style.overflow = 'hidden'; // Changed to hidden for canvas
    colDetail.style.padding = '0'; // Remove padding for canvas
    // colDetail.innerText = 'Detail View'; // Removed text
    mainContainer.appendChild(colDetail);

    // Bottom Container (2 columns)
    const bottomContainer = document.createElement('div');
    bottomContainer.style.height = '200px';
    bottomContainer.style.display = 'flex';
    bottomContainer.style.borderTop = `1px solid ${theme.border}`;
    document.body.appendChild(bottomContainer);

    // Column 4: Repl (50%)
    const colRepl = document.createElement('div');
    colRepl.style.flex = '0 0 50%';
    colRepl.style.borderRight = `1px solid ${theme.border}`;
    colRepl.style.padding = '10px';
    colRepl.style.display = 'flex';
    colRepl.style.flexDirection = 'column';
    bottomContainer.appendChild(colRepl);

    const replOutput = document.createElement('div');
    replOutput.style.flex = '1';
    replOutput.style.overflowY = 'auto';
    replOutput.style.fontFamily = 'monospace';
    replOutput.style.fontSize = '12px';
    replOutput.style.marginBottom = '5px';
    colRepl.appendChild(replOutput);

    const replInputContainer = document.createElement('div');
    replInputContainer.style.display = 'flex';
    colRepl.appendChild(replInputContainer);

    const replPrompt = document.createElement('span');
    replPrompt.innerText = '>>> ';
    replPrompt.style.marginRight = '5px';
    replPrompt.style.fontFamily = 'monospace';
    replInputContainer.appendChild(replPrompt);

    const replInput = document.createElement('input');
    replInput.style.flex = '1';
    replInput.style.backgroundColor = 'transparent';
    replInput.style.border = 'none';
    replInput.style.color = theme.inputFg;
    replInput.style.fontFamily = 'monospace';
    replInput.style.outline = 'none';
    replInputContainer.appendChild(replInput);

    // Column 5: Logs (50%)
    const colLogs = document.createElement('div');
    colLogs.style.flex = '0 0 50%';
    colLogs.style.overflowY = 'auto';
    colLogs.style.padding = '10px';
    colLogs.style.fontFamily = 'monospace';
    colLogs.style.fontSize = '12px';
    bottomContainer.appendChild(colLogs);

    // --- Logic ---

    function setStatus(text, color = theme.statusBarBg) {
        statusBar.innerText = text;
        statusBar.style.backgroundColor = color;
    }

    function addLog(text, color = theme.foreground) {
        const entry = document.createElement('div');
        entry.innerText = text;
        entry.style.color = color;
        entry.style.borderBottom = `1px solid ${theme.border}`;
        entry.style.padding = '2px 0';
        colLogs.appendChild(entry);
        colLogs.scrollTop = colLogs.scrollHeight;
    }

    function addToRepl(text, color = theme.foreground) {
        const entry = document.createElement('div');
        entry.innerText = text;
        entry.style.color = color;
        replOutput.appendChild(entry);
        replOutput.scrollTop = replOutput.scrollHeight;
    }

    function renderModelList(models) {
        colProperty.innerHTML = ''; // Clear
        const title = document.createElement('h3');
        title.innerText = 'Models';
        title.style.marginTop = '0';
        colProperty.appendChild(title);

        if (!models || models.length === 0) {
            colProperty.innerText += 'No models found.';
            return;
        }

        const table = document.createElement('table');
        table.style.width = '100%';
        table.style.borderCollapse = 'collapse';

        const headerRow = document.createElement('tr');
        ['ID', 'Docs', 'Name'].forEach(text => {
            const th = document.createElement('th');
            th.innerText = text;
            th.style.textAlign = 'left';
            th.style.borderBottom = `1px solid ${theme.border}`;
            th.style.padding = '5px';
            headerRow.appendChild(th);
        });
        table.appendChild(headerRow);

        models.forEach(model => {
            const row = document.createElement('tr');
            if (model.active) {
                row.style.backgroundColor = theme.selectionBg;
                // Load visualization for active model if not already loaded
                // But we only want to do this once or when it changes.
                // For now, let's rely on the user clicking or the initial load.
                // Actually, if it's active, we should probably load it if docData is null.
                if (!docData) {
                    loadVisualizationData(model.name, send);
                }
            }
            if (!model.enabled) {
                row.style.color = theme.secondary;
            }

            const tdId = document.createElement('td');
            tdId.innerText = model.id;
            tdId.style.padding = '5px';
            row.appendChild(tdId);

            const tdDocs = document.createElement('td');
            tdDocs.innerText = model.docs;
            tdDocs.style.padding = '5px';
            row.appendChild(tdDocs);

            const tdName = document.createElement('td');
            tdName.innerText = model.name + (model.enabled ? '' : ' (DISABLED)');
            tdName.style.padding = '5px';
            row.appendChild(tdName);

            row.style.cursor = 'pointer';
            row.addEventListener('click', () => {
                send('select', model.id);
                setStatus(`Selecting model ${model.name}...`, theme.logResponse);
                // Also trigger visualization load
                loadVisualizationData(model.name, send);
            });

            table.appendChild(row);
        });

        colProperty.appendChild(table);
    }

    function renderResultItem(res, index, showScore = true) {
        const container = document.createElement('div');
        container.style.marginBottom = '15px';
        container.style.borderBottom = `1px solid ${theme.border}`;
        container.style.paddingBottom = '10px';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';

        // Header Container
        const headerContainer = document.createElement('div');
        headerContainer.style.display = 'flex';
        headerContainer.style.marginBottom = '5px';
        container.appendChild(headerContainer);

        // Icon
        if (res.metadata && res.metadata.icon) {
            const icon = document.createElement('img');
            icon.src = `data:image/png;base64,${res.metadata.icon}`;
            icon.style.width = '48px';
            icon.style.height = '64px'; // Approx aspect ratio
            icon.style.objectFit = 'contain';
            icon.style.marginRight = '10px';
            icon.style.backgroundColor = theme.base3; // Icons might have transparent bg
            headerContainer.appendChild(icon);
        }

        // Info Block
        const infoBlock = document.createElement('div');
        infoBlock.style.flex = '1';
        infoBlock.style.display = 'flex';
        infoBlock.style.flexDirection = 'column';
        headerContainer.appendChild(infoBlock);

        // Line 1: ID, Score, Title
        const line1 = document.createElement('div');
        const titleText = (res.metadata && res.metadata.title) ? res.metadata.title : res.descriptor;
        const scoreHtml = showScore ? `<span style="color: ${theme.searchScore};">[${res.cosine.toFixed(3)}]</span> ` : '';
        const indexHtml = index !== null ? `<span style="color: ${theme.searchId}; font-weight: bold;">#${index + 1}</span> ` : '';

        line1.innerHTML = `${indexHtml}${scoreHtml}<span style="font-weight: bold; color: ${theme.searchTitle};">${titleText}</span>`;

        // Add Highlight Button
        if (docData && docData.pointMap) {
            const key = `${res.hash}:${res.chunk_index}`;
            if (docData.pointMap.hasOwnProperty(key)) {
                const pointIndex = docData.pointMap[key];
                const btn = document.createElement('button');
                btn.innerText = '⌖'; // Target icon
                btn.title = 'Locate in 3D';
                btn.style.marginLeft = '10px';
                btn.style.cursor = 'pointer';
                btn.style.border = 'none';
                btn.style.backgroundColor = 'transparent';
                btn.style.color = theme.primary;
                btn.onclick = (e) => {
                    e.stopPropagation();
                    highlightPoint(pointIndex);
                };
                line1.appendChild(btn);
            }
        }

        infoBlock.appendChild(line1);

        // Line 2: Authors
        if (res.metadata && res.metadata.authors && res.metadata.authors.length > 0) {
            const line2 = document.createElement('div');
            line2.style.fontSize = '0.9em';
            line2.style.color = theme.searchMeta;
            line2.style.fontStyle = 'italic';
            line2.innerText = res.metadata.authors.join(', ');
            infoBlock.appendChild(line2);
        }

        // Line 3: Descriptor
        const line3 = document.createElement('div');
        line3.style.fontSize = '0.8em';
        line3.style.color = theme.searchDesc;
        line3.style.fontFamily = 'monospace';
        line3.innerText = res.descriptor;
        infoBlock.appendChild(line3);

        // Text with Highlighting
        const textContainer = document.createElement('div');
        textContainer.style.marginTop = '5px';
        textContainer.style.fontSize = '12px';
        textContainer.style.lineHeight = '1.4';

        if (res.text) {
            if (res.significance) {
                let currentSpan = null;
                let currentSig = -1;

                for (let i = 0; i < res.text.length; i++) {
                    const char = res.text[i];
                    const sig = res.significance[i] || 0;

                    if (sig !== currentSig) {
                        if (currentSpan) {
                            textContainer.appendChild(currentSpan);
                        }
                        currentSpan = document.createElement('span');
                        if (sig > 0) {
                            const yellow = 255 - Math.min(sig * 255, 255);
                            currentSpan.style.backgroundColor = `rgb(255,255, ${yellow})`;
                            currentSpan.style.color = '#000';
                        }
                        currentSig = sig;
                    }
                    currentSpan.innerText += char;
                }
                if (currentSpan) {
                    textContainer.appendChild(currentSpan);
                }
            } else {
                textContainer.innerText = res.text;
            }
        }

        container.appendChild(textContainer);
        return container;
    }

    function renderSearchResults(results) {
        colMain.innerHTML = '';
        const title = document.createElement('h3');
        title.innerText = 'Search Results';
        title.style.marginTop = '0';
        colMain.appendChild(title);

        if (!results || results.length === 0) {
            colMain.innerText += 'No results found.';
            highlightDescriptors([]); // Clear highlights
            return;
        }

        results.forEach((res, index) => {
            const item = renderResultItem(res, index, true);
            colMain.appendChild(item);
        });

        // We no longer automatically highlight descriptors
        highlightDescriptors([]);
    }

    // Connect to WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);

    function send(cmd, payload) {
        const msg = {
            token: "dummy-token",
            uuid: crypto.randomUUID(),
            cmd: cmd,
            payload: payload
        };
        ws.send(JSON.stringify(msg));
    }
    sendRequest = send;

    replInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
            const commandLine = replInput.value.trim();
            if (commandLine) {
                addToRepl(`>>> ${commandLine}`);
                replInput.value = '';

                const parts = commandLine.split(' ');
                const cmd = parts[0];
                const args = parts.slice(1).join(' ');

                if (cmd === 'search') {
                    if (args) {
                        send('search', args);
                    } else {
                        addToRepl('Usage: search <query>', theme.error);
                    }
                } else if (cmd === 'list' && args === 'models') {
                    send('list_models', '');
                } else {
                    addToRepl(`Unknown command: ${cmd}`, theme.error);
                }
            }
        }
    });

    ws.onopen = function () {
        console.log("WebSocket connected");
        setStatus('Connected', theme.success);

        // Request model list on startup
        send("list_models", "");
    };

    let currentProgressBar = null;

    function createProgressBar() {
        const container = document.createElement('div');
        container.style.border = `1px solid ${theme.border}`;
        container.style.backgroundColor = theme.progressContainer;
        container.style.height = '18px';
        container.style.position = 'relative';
        container.style.marginBottom = '2px';

        const bar = document.createElement('div');
        bar.style.backgroundColor = theme.progressBar;
        bar.style.height = '100%';
        bar.style.width = '0%';
        bar.style.transition = 'width 0.1s ease';
        container.appendChild(bar);

        const label = document.createElement('div');
        label.style.position = 'absolute';
        label.style.top = '0';
        label.style.left = '5px';
        label.style.right = '5px';
        label.style.fontSize = '11px';
        label.style.lineHeight = '18px';
        label.style.color = theme.progressLabel;
        label.style.whiteSpace = 'nowrap';
        label.style.overflow = 'hidden';
        label.style.textOverflow = 'ellipsis';
        container.appendChild(label);

        replOutput.appendChild(container);
        replOutput.scrollTop = replOutput.scrollHeight;

        return { container, bar, label };
    }

    ws.onmessage = function (event) {
        try {
            const data = JSON.parse(event.data);
            const cmd = data.cmd;
            const payload = data.payload;
            const requestCmd = data.request_cmd;

            if (cmd === 'log') {
                addLog(`[LOG] ${payload}`, theme.logLog);
            } else if (cmd === 'progress') {
                console.log("Progress data:", data);
                let progressData = payload;
                if (typeof payload === 'string') {
                    try {
                        progressData = JSON.parse(payload);
                    } catch (e) {
                        console.error("Error parsing progress payload:", e);
                    }
                }

                const state = progressData.state || 'Processing...';
                const percentVal = progressData.percent_completion !== undefined ? progressData.percent_completion : 0;
                const percent = (percentVal * 100).toFixed(1);

                if (!currentProgressBar) {
                    currentProgressBar = createProgressBar();
                }

                currentProgressBar.bar.style.width = `${percent}%`;
                currentProgressBar.label.innerText = `${state} (${percent}%)`;

                if (progressData.finished) {
                    currentProgressBar = null;
                }

            } else if (cmd === 'response') {
                console.log("Response data:", data);
                const final = data.final ? ' [FINAL]' : '';
                // addLog(`[RESPONSE] ${requestCmd || ''}${final}`, theme.logResponse);

                if (requestCmd === 'list_models') {
                    renderModelList(payload);
                } else if (requestCmd === 'search') {
                    renderSearchResults(payload);
                    addToRepl(`Search completed. Found ${payload ? payload.length : 0} results.`, theme.searchId);
                } else if (requestCmd === 'select') {
                    if (payload.status === 'ok') {
                        setStatus(`Model ${payload.selected_id} selected`, theme.success);
                        send('list_models', ''); // Refresh list
                    } else {
                        setStatus(`Error selecting model: ${payload.error}`, theme.error);
                    }
                } else if (requestCmd === 'get_chunk_details') {
                    if (payload.error) {
                        if (infoDiv) infoDiv.innerText = 'Error loading details: ' + payload.error;
                    } else {
                        if (infoDiv) infoDiv.innerText = 'Details loaded.';
                        colMain.innerHTML = '';
                        const title = document.createElement('h3');
                        title.innerText = 'Selected Point';
                        title.style.marginTop = '0';
                        colMain.appendChild(title);
                        const item = renderResultItem(payload, null, false);
                        colMain.appendChild(item);
                    }
                } else if (requestCmd === 'get_3d_viz_data') {
                    if (payload.error) {
                        addLog(`Error loading 3D data: ${payload.error}`, theme.error);
                        if (infoDiv) infoDiv.innerText = `Error: ${payload.error}`;
                    } else {
                        createSceneFromData(payload);
                    }
                }
            } else {
                addLog(`[UNKNOWN] ${JSON.stringify(data)}`, theme.logUnknown);
            }
        } catch (e) {
            console.error("Error parsing message:", e);
            addLog(`[ERROR] Raw: ${event.data}`, theme.error);
        }
    };

    ws.onclose = function () {
        console.log("WebSocket disconnected");
        setStatus('Disconnected', theme.error);
    };

    ws.onerror = function (error) {
        console.error("WebSocket error:", error);
        setStatus('Error', theme.error);
    };

    // Initialize Visualization
    initVisualization(colDetail);
};
