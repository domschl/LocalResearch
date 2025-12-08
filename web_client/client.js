import * as THREE from './third_party/three/three.module.js';
import { OrbitControls } from './third_party/three/OrbitControls.js';

// --- Visualization Globals ---
let scene, camera, renderer, controls;
let pointsObject = null;
let raycaster, mouse;
let mouseDownPosition = new THREE.Vector2();
let selectionMarker = null;
let selectedPointIndex = null;
let originalColors = [];
let docData = null;
let animationFrameId = null;
let vizContainer = null;
let infoDiv = null;
let sendRequest = null;
let lastQuery = '';
let currentModelName = 'Unknown';

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
    renderer.domElement.addEventListener('mousedown', onMouseDown);
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

function onMouseDown(event) {
    mouseDownPosition.set(event.clientX, event.clientY);
}

function onMouseClick(event) {
    if (!docData || !pointsObject || !camera || !raycaster || !mouse) return;

    // Check if this was a drag (rotation) or a click
    const currentPos = new THREE.Vector2(event.clientX, event.clientY);
    if (currentPos.distanceTo(mouseDownPosition) > 5) {
        // Moved more than 5 pixels, treat as drag/rotation
        return;
    }

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
        if (selectionMarker) {
            scene.remove(selectionMarker);
            selectionMarker = null;
        }
    } else {
        // Dim all (simulate transparency by blending with background)
        // Background is 0xf0f0f0 (approx 0.94, 0.94, 0.94)
        const bgR = 0.94;
        const bgG = 0.94;
        const bgB = 0.94;
        const alpha = 0.3; // Low opacity for unselected

        for (let i = 0; i < colorsAttribute.count; i++) {
            const r = originalColors[i * 3];
            const g = originalColors[i * 3 + 1];
            const b = originalColors[i * 3 + 2];

            // lerp(bg, color, alpha) -> bg * (1-alpha) + color * alpha
            colorsAttribute.setXYZ(
                i,
                bgR * (1 - alpha) + r * alpha,
                bgG * (1 - alpha) + g * alpha,
                bgB * (1 - alpha) + b * alpha
            );
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
        }

        // Add marker
        const positions = pointsObject.geometry.attributes.position;
        const x = positions.getX(pointIndex);
        const y = positions.getY(pointIndex);
        const z = positions.getZ(pointIndex);
        addSelectionMarker(new THREE.Vector3(x, y, z));
    }
    colorsAttribute.needsUpdate = true;
}

function addSelectionMarker(position) {
    if (selectionMarker) {
        scene.remove(selectionMarker);
    }

    const geometry = new THREE.SphereGeometry(0.1, 8, 8);
    const material = new THREE.MeshBasicMaterial({
        color: 0xff0000,
        wireframe: true,
        transparent: true,
        opacity: 0.8
    });
    selectionMarker = new THREE.Mesh(geometry, material);
    selectionMarker.position.copy(position);
    scene.add(selectionMarker);
}

function focusOnPoint(pointIndex) {
    if (!pointsObject || !pointsObject.geometry) return;

    // Highlight first
    highlightPoint(pointIndex);

    const positions = pointsObject.geometry.attributes.position;
    const x = positions.getX(pointIndex);
    const y = positions.getY(pointIndex);
    const z = positions.getZ(pointIndex);
    const target = new THREE.Vector3(x, y, z);

    // Animate controls target
    // For simplicity, just set it. We could tween it for smoothness.
    if (controls) {
        controls.target.copy(target);

        // Move camera if it's too far? Or just let user zoom.
        // Let's move it slightly closer if it's very far, but maintain direction.
        const dist = camera.position.distanceTo(target);
        if (dist > 5) {
            const dir = new THREE.Vector3().subVectors(camera.position, target).normalize();
            camera.position.copy(target).add(dir.multiplyScalar(3));
        }

        controls.update();
    }
}

function highlightDescriptors(descriptors) {
    if (!pointsObject || !pointsObject.geometry || !docData) return;

    const colorsAttribute = pointsObject.geometry.attributes.color;

    if (!descriptors || descriptors.length === 0) {
        // Restore all to original colors
        for (let i = 0; i < colorsAttribute.count; i++) {
            colorsAttribute.setXYZ(i, originalColors[i * 3], originalColors[i * 3 + 1], originalColors[i * 3 + 2]);
        }
        colorsAttribute.needsUpdate = true;
        return;
    }

    // Dim all first (simulate transparency)
    const bgR = 0.94;
    const bgG = 0.94;
    const bgB = 0.94;
    const alpha = 0.3;

    for (let i = 0; i < colorsAttribute.count; i++) {
        const r = originalColors[i * 3];
        const g = originalColors[i * 3 + 1];
        const b = originalColors[i * 3 + 2];

        colorsAttribute.setXYZ(
            i,
            bgR * (1 - alpha) + r * alpha,
            bgG * (1 - alpha) + g * alpha,
            bgB * (1 - alpha) + b * alpha
        );
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
        if (selectionMarker) {
            scene.remove(selectionMarker);
            selectionMarker = null;
        }
        selectedPointIndex = null;
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

        /* Pane System */
        .pane {
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
            background-color: ${theme.background};
        }
        .pane-title {
            height: 24px;
            background-color: ${theme.base2};
            color: ${theme.base01};
            display: flex;
            align-items: center;
            padding: 0 8px;
            font-size: 11px;
            font-weight: bold;
            border-bottom: 1px solid ${theme.border};
            user-select: none;
            flex-shrink: 0;
        }
        .pane-title .title-text {
            flex: 1;
        }
        .pane-title .pane-controls {
            display: flex;
            gap: 4px;
        }
        .pane-control-btn {
            cursor: pointer;
            width: 16px;
            height: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 2px;
        }
        .pane-control-btn:hover {
            background-color: ${theme.base1};
            color: ${theme.base3};
        }
        .pane-content {
            flex: 1;
            overflow: auto;
            position: relative;
        }
        
        /* Splitters */
        .splitter {
            background-color: ${theme.border};
            z-index: 10;
        }
        .splitter:hover, .splitter.active {
            background-color: ${theme.primary};
        }
        .splitter-v {
            width: 4px;
            cursor: col-resize;
            flex-shrink: 0;
        }
        .splitter-h {
            height: 4px;
            cursor: row-resize;
            flex-shrink: 0;
        }
        
        /* Toolbar icons */
        .toolbar-icon {
            cursor: pointer;
            margin-left: 8px;
            opacity: 0.7;
        }
        .toolbar-icon:hover {
            opacity: 1;
        }
        .toolbar-icon.inactive {
            opacity: 0.3;
            text-decoration: line-through;
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
    document.body.appendChild(statusBar);

    const statusText = document.createElement('span');
    statusText.innerText = 'Connecting...';
    statusText.style.flex = '1';
    statusBar.appendChild(statusText);

    // --- Pane Management ---
    const panes = {};
    const splitters = [];

    function createPane(id, title, parent, initialFlex, contentElement = null) {
        const pane = document.createElement('div');
        pane.className = 'pane';
        pane.id = `pane-${id}`;
        // Parse initialFlex to get basis if possible, or just use it.
        // We want flex-grow: 1, flex-shrink: 1, flex-basis: initialFlex
        // If initialFlex is "0 0 20%", we want "1 1 20%".
        // Let's assume initialFlex passed is the basis or full string.
        // The caller passes "0 0 20%". We should change that.
        // Let's just override here.
        const basis = initialFlex.split(' ')[2] || initialFlex || 'auto'; // Handle cases where initialFlex is just a basis string
        pane.style.flex = `1 1 ${basis}`;

        // Title Bar
        const titleBar = document.createElement('div');
        titleBar.className = 'pane-title';

        const titleText = document.createElement('span');
        titleText.className = 'title-text';
        titleText.innerText = title;
        titleBar.appendChild(titleText);

        const controls = document.createElement('div');
        controls.className = 'pane-controls';

        const hideBtn = document.createElement('div');
        hideBtn.className = 'pane-control-btn';
        hideBtn.innerText = '×';
        hideBtn.title = 'Hide Pane';
        hideBtn.onclick = () => togglePane(id, false);
        controls.appendChild(hideBtn);

        titleBar.appendChild(controls);
        pane.appendChild(titleBar);

        // Content Area
        const content = document.createElement('div');
        content.className = 'pane-content';
        if (contentElement) {
            content.appendChild(contentElement);
        }
        pane.appendChild(content);

        parent.appendChild(pane);

        panes[id] = {
            element: pane,
            parent: parent,
            initialFlex: initialFlex,
            visible: true,
            title: title
        };

        return { pane, content };
    }

    function createSplitter(parent, orientation, prevPaneId, nextPaneId) {
        const splitter = document.createElement('div');
        splitter.className = `splitter splitter-${orientation === 'horizontal' ? 'h' : 'v'}`;
        parent.appendChild(splitter);

        let isDragging = false;

        splitter.addEventListener('mousedown', (e) => {
            isDragging = true;
            splitter.classList.add('active');
            document.body.style.cursor = orientation === 'horizontal' ? 'row-resize' : 'col-resize';
            e.preventDefault();
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const prevPane = panes[prevPaneId].element;
            const nextPane = panes[nextPaneId].element;

            if (!panes[prevPaneId].visible || !panes[nextPaneId].visible) return;

            const parentRect = parent.getBoundingClientRect();
            const prevRect = prevPane.getBoundingClientRect();
            const nextRect = nextPane.getBoundingClientRect();

            if (orientation === 'horizontal') {
                // Row resizing (height)
                // Calculate total available height for these two panes
                const totalHeight = prevRect.height + nextRect.height;
                const startY = prevRect.top;

                // New height for prevPane
                let newPrevHeight = e.clientY - startY;

                // Constrain
                if (newPrevHeight < 50) newPrevHeight = 50;
                if (newPrevHeight > totalHeight - 50) newPrevHeight = totalHeight - 50;

                const newNextHeight = totalHeight - newPrevHeight;

                const prevPercent = (newPrevHeight / parentRect.height) * 100;
                const nextPercent = (newNextHeight / parentRect.height) * 100;

                prevPane.style.flexBasis = `${prevPercent}%`;
                nextPane.style.flexBasis = `${nextPercent}%`;

            } else {
                // Column resizing (width)
                const totalWidth = prevRect.width + nextRect.width;
                const startX = prevRect.left;

                let newPrevWidth = e.clientX - startX;

                if (newPrevWidth < 50) newPrevWidth = 50;
                if (newPrevWidth > totalWidth - 50) newPrevWidth = totalWidth - 50;

                const newNextWidth = totalWidth - newPrevWidth;

                const prevPercent = (newPrevWidth / parentRect.width) * 100;
                const nextPercent = (newNextWidth / parentRect.width) * 100;

                prevPane.style.flexBasis = `${prevPercent}%`;
                nextPane.style.flexBasis = `${nextPercent}%`;

                // Trigger resize for 3D view
                if (typeof onWindowResize === 'function') onWindowResize();
            }
        });

        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                splitter.classList.remove('active');
                document.body.style.cursor = '';
            }
        });

        splitters.push({ element: splitter, prev: prevPaneId, next: nextPaneId });
        return splitter;
    }

    function togglePane(id, show) {
        const p = panes[id];
        if (!p) return;

        p.visible = show;
        p.element.style.display = show ? 'flex' : 'none';

        updateSplitters();

        const icon = document.getElementById(`toggle-icon-${id}`);
        if (icon) {
            if (show) icon.classList.remove('inactive');
            else icon.classList.add('inactive');
        }
    }

    function updateSplitters() {
        splitters.forEach(s => {
            const prev = panes[s.prev];
            const next = panes[s.next];
            if (prev.visible && next.visible) {
                s.element.style.display = 'block';
            } else {
                s.element.style.display = 'none';
            }
        });
    }

    // --- Layout Construction ---

    const workspace = document.createElement('div');
    workspace.style.flex = '1';
    workspace.style.display = 'flex';
    workspace.style.flexDirection = 'column';
    workspace.style.overflow = 'hidden';
    document.body.appendChild(workspace);

    // Upper Area (Columns)
    const upperArea = document.createElement('div');
    upperArea.style.display = 'flex';
    upperArea.style.flex = '1';
    upperArea.style.overflow = 'hidden';
    workspace.appendChild(upperArea);

    // Column 1: Property
    const colPropertyContent = document.createElement('div');
    colPropertyContent.style.height = '100%';
    colPropertyContent.style.overflowY = 'auto';
    colPropertyContent.style.padding = '10px';

    const { content: colPropertyWrapper } = createPane('property', 'Models', upperArea, '0 0 20%', colPropertyContent);
    const colProperty = colPropertyContent;

    createSplitter(upperArea, 'vertical', 'property', 'main');

    // Column 2: Main
    const colMainContent = document.createElement('div');
    colMainContent.style.height = '100%';
    colMainContent.style.overflowY = 'auto';
    colMainContent.style.padding = '10px';
    colMainContent.innerText = 'Main Content Area';
    colMainContent.style.backgroundColor = theme.background;

    createPane('main', 'Search & Content', upperArea, '0 0 40%', colMainContent);
    const colMain = colMainContent;

    createSplitter(upperArea, 'vertical', 'main', 'detail');

    // Column 3: Detail
    const colDetailContent = document.createElement('div');
    colDetailContent.style.height = '100%';
    colDetailContent.style.overflow = 'hidden';
    colDetailContent.style.position = 'relative';
    colDetailContent.style.padding = '0';

    createPane('detail', 'Visualization', upperArea, '0 0 40%', colDetailContent);
    const colDetail = colDetailContent;

    // Horizontal Splitter between Upper and Lower
    const hSplitter = document.createElement('div');
    hSplitter.className = 'splitter splitter-h';
    workspace.appendChild(hSplitter);

    // Lower Area
    const lowerArea = document.createElement('div');
    lowerArea.style.display = 'flex';
    lowerArea.style.height = '200px';
    lowerArea.style.overflow = 'hidden';
    workspace.appendChild(lowerArea);

    // Resize logic for H-Splitter
    let hDragging = false;
    hSplitter.addEventListener('mousedown', (e) => {
        hDragging = true;
        hSplitter.classList.add('active');
        document.body.style.cursor = 'row-resize';
        e.preventDefault();
    });
    document.addEventListener('mousemove', (e) => {
        if (!hDragging) return;
        const totalHeight = workspace.clientHeight;
        const newLowerHeight = totalHeight - (e.clientY - workspace.getBoundingClientRect().top);
        if (newLowerHeight > 50 && newLowerHeight < totalHeight - 50) {
            lowerArea.style.height = `${newLowerHeight}px`;
            // Trigger resize for 3D view
            if (typeof onWindowResize === 'function') onWindowResize();
        }
    });
    document.addEventListener('mouseup', () => {
        if (hDragging) {
            hDragging = false;
            hSplitter.classList.remove('active');
            document.body.style.cursor = '';
        }
    });

    // Lower Columns
    const colReplContent = document.createElement('div');
    colReplContent.style.display = 'flex';
    colReplContent.style.flexDirection = 'column';
    colReplContent.style.height = '100%';
    colReplContent.style.padding = '10px';

    createPane('repl', 'REPL', lowerArea, '0 0 50%', colReplContent);
    const colRepl = colReplContent;

    createSplitter(lowerArea, 'vertical', 'repl', 'logs');

    const colLogsContent = document.createElement('div');
    colLogsContent.style.height = '100%';
    colLogsContent.style.overflowY = 'auto';
    colLogsContent.style.padding = '10px';
    colLogsContent.style.fontFamily = 'monospace';
    colLogsContent.style.fontSize = '12px';

    createPane('logs', 'Logs', lowerArea, '0 0 50%', colLogsContent);
    const colLogs = colLogsContent;

    // Repl Internals
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

    // Top Bar Controls
    const toolbar = document.createElement('div');
    toolbar.style.marginLeft = 'auto';
    toolbar.style.display = 'flex';

    ['property', 'main', 'detail', 'repl', 'logs'].forEach(id => {
        const icon = document.createElement('span');
        icon.className = 'toolbar-icon';
        icon.id = `toggle-icon-${id}`;
        icon.innerText = id.charAt(0).toUpperCase();
        icon.title = `Toggle ${id}`;
        icon.onclick = () => {
            const p = panes[id];
            togglePane(id, !p.visible);
        };
        toolbar.appendChild(icon);
    });

    statusBar.appendChild(toolbar);
    // --- Logic ---

    function setStatus(text, color = theme.statusBarBg) {
        statusText.innerText = text;
        statusBar.style.backgroundColor = color;
    }

    function addLog(text, color = theme.foreground) {
        const entry = document.createElement('div');
        entry.innerText = text;
        entry.style.color = color;
        entry.style.borderBottom = `1px solid ${theme.border} `;
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
            th.style.borderBottom = `1px solid ${theme.border} `;
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
                currentModelName = model.name;
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

    function renderResultItem(res, index, showScore = true, onClose = null) {
        const container = document.createElement('div');
        container.style.marginBottom = '15px';
        container.style.borderBottom = `1px solid ${theme.border} `;
        container.style.paddingBottom = '10px';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';

        // Header Container
        const headerContainer = document.createElement('div');
        headerContainer.style.display = 'flex';
        headerContainer.style.marginBottom = '5px';
        container.appendChild(headerContainer);

        // Close Button (if provided)
        if (onClose) {
            const closeBtn = document.createElement('button');
            closeBtn.innerText = '×';
            closeBtn.title = 'Close this result';
            closeBtn.style.border = 'none';
            closeBtn.style.background = 'transparent';
            closeBtn.style.color = theme.secondary;
            closeBtn.style.cursor = 'pointer';
            closeBtn.style.fontSize = '16px';
            closeBtn.style.marginLeft = 'auto'; // Push to right
            closeBtn.style.alignSelf = 'flex-start';
            closeBtn.onclick = (e) => {
                e.stopPropagation();
                onClose();
            };
            headerContainer.appendChild(closeBtn);
        }

        // Icon
        if (res.metadata && res.metadata.icon) {
            const icon = document.createElement('img');
            icon.src = `data: image / png; base64, ${res.metadata.icon} `;
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
        const scoreHtml = showScore ? `<span style="color: ${theme.searchScore};">[${res.cosine.toFixed(3)}]</span>` : '';
        const indexHtml = index !== null ? `<span style="color: ${theme.searchId}; font-weight: bold;">#${index + 1}</span>` : '';

        line1.innerHTML = `${indexHtml} ${scoreHtml} <span style="font-weight: bold; color: ${theme.searchTitle};">${titleText}</span>`;

        // Add Highlight Button
        if (docData && docData.pointMap) {
            const key = `${res.hash}:${res.chunk_index} `;
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
                    focusOnPoint(pointIndex);
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
                            currentSpan.style.backgroundColor = `rgb(255, 255, ${yellow})`;
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

    function renderSearchResults(results, query, modelName) {
        // colMain.innerHTML = ''; // Do NOT clear

        if (!results || results.length === 0) {
            addToRepl('No results found.', theme.warning);
            highlightDescriptors([]);
            return;
        }

        const blockContainer = document.createElement('div');
        blockContainer.style.marginBottom = '20px';
        blockContainer.style.border = `1px solid ${theme.border} `;
        blockContainer.style.borderRadius = '5px';
        blockContainer.style.backgroundColor = theme.background;

        // Header
        const header = document.createElement('div');
        header.style.padding = '10px';
        header.style.backgroundColor = theme.base2;
        header.style.borderBottom = `1px solid ${theme.border} `;
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';

        const title = document.createElement('span');
        title.innerHTML = `Search: <strong>${query}</strong> <span style="font-size: 0.8em; color: ${theme.secondary}">(${modelName})</span>`;
        header.appendChild(title);

        const closeAllBtn = document.createElement('button');
        closeAllBtn.innerText = 'Close Search';
        closeAllBtn.style.border = 'none';
        closeAllBtn.style.background = 'transparent';
        closeAllBtn.style.color = theme.red;
        closeAllBtn.style.cursor = 'pointer';
        closeAllBtn.onclick = () => {
            blockContainer.remove();
        };
        header.appendChild(closeAllBtn);
        blockContainer.appendChild(header);

        // Results
        const resultsContainer = document.createElement('div');
        resultsContainer.style.padding = '10px';
        blockContainer.appendChild(resultsContainer);

        results.forEach((res, index) => {
            const item = renderResultItem(res, index, true, () => {
                item.remove();
                // If no items left, maybe remove the block? 
                if (resultsContainer.children.length === 0) {
                    blockContainer.remove();
                }
            });
            resultsContainer.appendChild(item);
        });

        colMain.appendChild(blockContainer);
        colMain.scrollTop = colMain.scrollHeight; // Auto scroll to bottom

        highlightDescriptors([]);
    }

    function renderChunkDetails(chunk) {
        const blockContainer = document.createElement('div');
        blockContainer.style.marginBottom = '20px';
        blockContainer.style.border = `1px solid ${theme.border} `;
        blockContainer.style.borderRadius = '5px';
        blockContainer.style.backgroundColor = theme.background;

        // Header
        const header = document.createElement('div');
        header.style.padding = '10px';
        header.style.backgroundColor = theme.selectionBg; // Distinct color for selection
        header.style.borderBottom = `1px solid ${theme.border} `;
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';

        const title = document.createElement('span');
        title.innerHTML = `Selected Point < span style = "font-size: 0.8em; color: ${theme.secondary}" > (${currentModelName})</span > `;
        header.appendChild(title);

        const closeBtn = document.createElement('button');
        closeBtn.innerText = 'Close';
        closeBtn.style.border = 'none';
        closeBtn.style.background = 'transparent';
        closeBtn.style.color = theme.red;
        closeBtn.style.cursor = 'pointer';
        closeBtn.onclick = () => {
            blockContainer.remove();
        };
        header.appendChild(closeBtn);
        blockContainer.appendChild(header);

        // Content
        const contentContainer = document.createElement('div');
        contentContainer.style.padding = '10px';
        blockContainer.appendChild(contentContainer);

        const item = renderResultItem(chunk, null, false, () => {
            blockContainer.remove();
        });
        contentContainer.appendChild(item);

        colMain.appendChild(blockContainer);
        colMain.scrollTop = colMain.scrollHeight;
    }

    function renderTimeline(events, criteria) {
        const blockContainer = document.createElement('div');
        blockContainer.style.marginBottom = '20px';
        blockContainer.style.border = `1px solid ${theme.border} `;
        blockContainer.style.borderRadius = '5px';
        blockContainer.style.backgroundColor = theme.background;

        // Header
        const header = document.createElement('div');
        header.style.padding = '10px';
        header.style.backgroundColor = theme.base2;
        header.style.borderBottom = `1px solid ${theme.border} `;
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';

        const title = document.createElement('span');
        let criteriaText = [];
        if (criteria.time) criteriaText.push(`Time: ${criteria.time} `);
        if (criteria.domains) criteriaText.push(`Domains: ${criteria.domains} `);
        if (criteria.keywords) criteriaText.push(`Keywords: ${criteria.keywords} `);

        title.innerHTML = `Timeline: <strong>${criteriaText.join(', ') || 'All'}</strong>`;
        header.appendChild(title);

        const closeBtn = document.createElement('button');
        closeBtn.innerText = 'Close';
        closeBtn.style.border = 'none';
        closeBtn.style.background = 'transparent';
        closeBtn.style.color = theme.red;
        closeBtn.style.cursor = 'pointer';
        closeBtn.onclick = () => {
            blockContainer.remove();
        };
        header.appendChild(closeBtn);
        blockContainer.appendChild(header);

        // Content
        const contentContainer = document.createElement('div');
        contentContainer.style.padding = '10px';
        contentContainer.style.maxHeight = '400px';
        contentContainer.style.overflowY = 'auto';
        blockContainer.appendChild(contentContainer);

        if (!events || events.length === 0) {
            contentContainer.innerText = "No events found.";
        } else {
            const table = document.createElement('table');
            table.style.width = '100%';
            table.style.borderCollapse = 'collapse';

            const thead = document.createElement('tr');
            ['Date', 'Event'].forEach(text => {
                const th = document.createElement('th');
                th.innerText = text;
                th.style.textAlign = 'left';
                th.style.borderBottom = `1px solid ${theme.border} `;
                th.style.padding = '5px';
                thead.appendChild(th);
            });
            table.appendChild(thead);

            events.forEach(evt => {
                const row = document.createElement('tr');

                const tdDate = document.createElement('td');
                tdDate.innerText = evt.date;
                tdDate.style.padding = '5px';
                tdDate.style.borderBottom = `1px solid ${theme.border} 40`; // Faint border
                tdDate.style.whiteSpace = 'nowrap';
                tdDate.style.verticalAlign = 'top';
                tdDate.style.color = theme.blue;
                row.appendChild(tdDate);

                const tdEvent = document.createElement('td');
                tdEvent.innerText = evt.event;
                tdEvent.style.padding = '5px';
                tdEvent.style.borderBottom = `1px solid ${theme.border} 40`;
                row.appendChild(tdEvent);

                table.appendChild(row);
            });
            contentContainer.appendChild(table);
        }

        colMain.appendChild(blockContainer);
        colMain.scrollTop = colMain.scrollHeight;
    }

    // Connect to WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);

    function generateUUID() {
        if (typeof crypto !== 'undefined' && crypto.randomUUID) {
            return crypto.randomUUID();
        }
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    function send(cmd, payload) {
        const msg = {
            token: "dummy-token",
            uuid: generateUUID(),
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

                const parts = commandLine.split(' ');
                const cmd = parts[0];
                const args = parts.slice(1).join(' ');

                if (cmd === 'search') {
                    if (args) {
                        console.log("Sending search:", args);
                        lastQuery = args;
                        send('search', args);
                    } else {
                        addToRepl('Usage: search <query>', theme.error);
                    }
                } else if (cmd === 'list' && args === 'models') {
                    send('list_models', '');
                } else if (cmd === 'timeline') {
                    // Parse args: time=... domains=... keywords=...
                    // Simple parser
                    const payload = {};
                    const regex = /(\w+)=("[^"]*"|\S+)/g;
                    let match;
                    while ((match = regex.exec(args)) !== null) {
                        let val = match[2];
                        if (val.startsWith('"') && val.endsWith('"')) {
                            val = val.slice(1, -1);
                        }
                        payload[match[1]] = val;
                    }
                    // If no named args but just text, assume keywords? Or just send empty if no args?
                    // The CLI uses a specific parser. Here we just support named args for now as per request.
                    // If args is present but regex didn't match anything, maybe it's just keywords?
                    // But user request example: timeline [time=...]

                    lastQuery = payload; // Store for display
                    send('timeline', payload);
                    addToRepl(`Requesting timeline...`, theme.logLog);
                } else {
                    addToRepl(`Unknown command: ${cmd} `, theme.error);
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
        container.style.border = `1px solid ${theme.border} `;
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
                addLog(`[LOG] ${payload} `, theme.logLog);
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

                currentProgressBar.bar.style.width = `${percent}% `;
                currentProgressBar.label.innerText = `${state} (${percent}%)`;

                if (progressData.finished) {
                    currentProgressBar = null;
                }

            } else if (cmd === 'response') {
                console.log("Response data:", data);
                const final = data.final ? ' [FINAL]' : '';
                // addLog(`[RESPONSE] ${ requestCmd || '' }${ final } `, theme.logResponse);

                if (requestCmd === 'list_models') {
                    renderModelList(payload);
                } else if (requestCmd === 'search') {
                    console.log("Search Payload:", payload);
                    renderSearchResults(payload, lastQuery, currentModelName);
                    addToRepl(`Search completed.Found ${payload ? payload.length : 0} results.`, theme.searchId);
                } else if (requestCmd === 'select') {
                    if (payload.status === 'ok') {
                        setStatus(`Model ${payload.selected_id} selected`, theme.success);
                        // currentModelName is updated in renderModelList when we refresh
                        send('list_models', ''); // Refresh list
                    } else {
                        setStatus(`Error selecting model: ${payload.error} `, theme.error);
                    }
                } else if (requestCmd === 'get_chunk_details') {
                    if (payload.error) {
                        if (infoDiv) infoDiv.innerText = 'Error loading details: ' + payload.error;
                    } else {
                        if (infoDiv) infoDiv.innerText = 'Details loaded.';
                        renderChunkDetails(payload);
                    }
                } else if (requestCmd === 'timeline') {
                    if (payload.error) {
                        addToRepl(`Error loading timeline: ${payload.error} `, theme.error);
                    } else {
                        renderTimeline(payload, lastQuery || {});
                        addToRepl(`Timeline loaded.${payload.length} events.`, theme.success);
                    }
                } else if (requestCmd === 'get_3d_viz_data') {
                    if (payload.error) {
                        addLog(`Error loading 3D data: ${payload.error} `, theme.error);
                        if (infoDiv) infoDiv.innerText = `Error: ${payload.error} `;
                    } else {
                        createSceneFromData(payload);
                    }
                }
            } else {
                addLog(`[UNKNOWN] ${JSON.stringify(data)} `, theme.logUnknown);
            }
        } catch (e) {
            console.error("Error parsing message:", e);
            addLog(`[ERROR] Raw: ${event.data} `, theme.error);
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
