import * as THREE from '/js/three.module.js';
import { OrbitControls } from '/js/OrbitControls.js';

// Global variables
let scene, camera, renderer, controls;
let pointsObject, connectionLines = [];
let raycaster, mouse;
let selectedPoint = null;
let originalColors = [];
let statsElement;
let docData;

// Initialize the scene
init();

function init() {
    // Setup renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.body.appendChild(renderer.domElement);
    
    // Setup scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    // Setup camera
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;
    
    // Setup controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 0);
    scene.add(directionalLight);
    
    // Raycaster for point selection
    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.1;
    mouse = new THREE.Vector2();
    
    // Stats display
    statsElement = document.getElementById('stats');
    
    // Initialize UI controls
    document.getElementById('resetView').addEventListener('click', resetView);
    document.getElementById('pointSize').addEventListener('input', updatePointSize);
    document.getElementById('showConnections').addEventListener('change', toggleConnections);
    
    // Load data from the API
    loadData();
}

function loadData() {
    const loadingElement = document.getElementById('loading');
    const progressBar = document.querySelector('.progress-bar');
    
    // Get URL parameters to see if a specific model is requested
    const urlParams = new URLSearchParams(window.location.search);
    const modelParam = urlParams.get('model');
    const maxPoints = urlParams.get('max');
    
    // API endpoint
    let dataUrl = '/api/embeddings';
    const params = [];
    
    if (modelParam) params.push(`model=${modelParam}`);
    if (maxPoints) params.push(`max=${maxPoints}`);
    
    if (params.length > 0) {
        dataUrl += '?' + params.join('&');
    }
    
    fetch(dataUrl)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // Check if content is large and show progress
            const contentLength = response.headers.get('Content-Length');
            if (contentLength && parseInt(contentLength) > 10000000) {
                const reader = response.body.getReader();
                const totalLength = parseInt(contentLength);
                let receivedLength = 0;
                
                loadingElement.querySelector('div').textContent = 'Loading large dataset...';
                
                return new ReadableStream({
                    start(controller) {
                        const push = () => {
                            reader.read().then(({ done, value }) => {
                                if (done) {
                                    controller.close();
                                    return;
                                }
                                
                                receivedLength += value.length;
                                const percentage = (receivedLength / totalLength * 100).toFixed(0);
                                progressBar.style.width = percentage + '%';
                                loadingElement.querySelector('div').textContent = 
                                    `Loading large dataset... ${percentage}%`;
                                
                                controller.enqueue(value);
                                push();
                            });
                        }
                        push();
                    }
                });
            }
            return response.body;
        })
        .then(body => {
            if (body instanceof ReadableStream) {
                return new Response(body).json();
            }
            return response.json();
        })
        .then(data => {
            docData = data;
            loadingElement.querySelector('div').textContent = 'Processing visualization...';
            
            // Process on next frame to allow UI update
            setTimeout(() => {
                try {
                    createVisualization(data);
                    loadingElement.style.display = 'none';
                    updateStats();
                } catch (error) {
                    loadingElement.innerHTML = `<div class="error">Error creating visualization: ${error.message}</div>`;
                    console.error('Visualization error:', error);
                }
            }, 10);
        })
        .catch(error => {
            loadingElement.innerHTML = `<div class="error">Error loading data: ${error.message}</div>`;
            console.error('Error loading data:', error);
        });
}

function createVisualization(data) {
    const positions = data.points;
    const colors = data.colors;
    
    // Calculate bounds for normalization
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    
    for (const pos of positions) {
        minX = Math.min(minX, pos[0]);
        maxX = Math.max(maxX, pos[0]);
        minY = Math.min(minY, pos[1]);
        maxY = Math.max(maxY, pos[1]);
        minZ = Math.min(minZ, pos[2]);
        maxZ = Math.max(maxZ, pos[2]);
    }
    
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    const rangeZ = maxZ - minZ;
    const maxRange = Math.max(rangeX, rangeY, rangeZ);
    
    // Create geometry
    const geometry = new THREE.BufferGeometry();
    const normalizedPositions = [];
    
    for (const pos of positions) {
        // Normalize to range -5 to 5
        const x = ((pos[0] - minX) / maxRange * 10) - 5;
        const y = ((pos[1] - minY) / maxRange * 10) - 5;
        const z = ((pos[2] - minZ) / maxRange * 10) - 5;
        normalizedPositions.push(x, y, z);
    }
    
    // Create flat array of colors
    const colorArray = new Float32Array(colors.flat());
    originalColors = [...colorArray];
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(normalizedPositions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colorArray, 3));
    
    // Create points material
    const material = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: true,
        sizeAttenuation: true,
    });
    
    // Create points object
    pointsObject = new THREE.Points(geometry, material);
    scene.add(pointsObject);
    
    // Create document groupings (only if fewer than 50k points for performance)
    if (positions.length < 50000) {
        createDocumentConnections(data, normalizedPositions);
    } else {
        document.getElementById('showConnections').checked = false;
        document.getElementById('showConnections').disabled = true;
        document.getElementById('showConnections').parentElement.title = 
            "Disabled for large datasets (>50k points)";
    }
    
    // Add event listeners
    window.addEventListener('click', onMouseClick);
    window.addEventListener('resize', onWindowResize);
    
    // Start animation loop
    animate();
}

function createDocumentConnections(data, normalizedPositions) {
    // Group points by document
    const docGroups = {};
    
    for (let i = 0; i < data.docs.length; i++) {
        const docId = data.docs[i];
        if (!docGroups[docId]) {
            docGroups[docId] = [];
        }
        docGroups[docId].push(i);
    }
    
    // For each document with multiple chunks, create connections
    for (const docId in docGroups) {
        const indices = docGroups[docId];
        if (indices.length > 1) {
            // Calculate centroid
            const centroid = [0, 0, 0];
            for (const idx of indices) {
                centroid[0] += normalizedPositions[idx * 3];
                centroid[1] += normalizedPositions[idx * 3 + 1];
                centroid[2] += normalizedPositions[idx * 3 + 2];
            }
            centroid[0] /= indices.length;
            centroid[1] /= indices.length;
            centroid[2] /= indices.length;
            
            // Create connections (limit for performance)
            const connectionLimit = Math.min(indices.length, 10);
            const docColor = new THREE.Color(
                data.colors[indices[0]][0],
                data.colors[indices[0]][1],
                data.colors[indices[0]][2]
            );
            
            for (let i = 0; i < connectionLimit; i++) {
                const idx = indices[i];
                const lineGeometry = new THREE.BufferGeometry();
                const linePositions = [
                    normalizedPositions[idx * 3],
                    normalizedPositions[idx * 3 + 1],
                    normalizedPositions[idx * 3 + 2],
                    centroid[0], centroid[1], centroid[2]
                ];
                
                lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
                
                const lineMaterial = new THREE.LineBasicMaterial({
                    color: docColor,
                    transparent: true,
                    opacity: 0.2
                });
                
                const line = new THREE.Line(lineGeometry, lineMaterial);
                scene.add(line);
                connectionLines.push(line);
            }
        }
    }
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onMouseClick(event) {
    // Calculate mouse position in normalized device coordinates
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    
    // Update the picking ray with the camera and mouse position
    raycaster.setFromCamera(mouse, camera);
    
    // Check for intersections
    if (pointsObject) {
        const intersects = raycaster.intersectObject(pointsObject);
        
        if (intersects.length > 0) {
            const index = intersects[0].index;
            
            // Show information about the selected point
            const infoDiv = document.getElementById('info');
            infoDiv.innerHTML = `
                <strong>Document:</strong> ${docData.docs[index]}<br>
                <strong>Text:</strong> ${docData.texts[index]}
            `;
            
            // Highlight the selected point and its document
            highlightPoint(index);
        }
    }
}

function highlightPoint(index) {
    // Reset previous highlights
    if (selectedPoint !== null) {
        const colors = pointsObject.geometry.attributes.color;
        for (let i = 0; i < colors.count; i++) {
            colors.setXYZ(
                i,
                originalColors[i * 3],
                originalColors[i * 3 + 1],
                originalColors[i * 3 + 2]
            );
        }
    }
    
    // Highlight the selected document
    const docId = docData.docs[index];
    const colors = pointsObject.geometry.attributes.color;
    
    for (let i = 0; i < docData.docs.length; i++) {
        if (docData.docs[i] === docId) {
            // Highlight with yellow
            colors.setXYZ(i, 1.0, 1.0, 0.0);
        }
    }
    
    colors.needsUpdate = true;
    selectedPoint = index;
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function resetView() {
    camera.position.set(0, 0, 5);
    controls.reset();
}

function updatePointSize(event) {
    if (pointsObject) {
        pointsObject.material.size = parseFloat(event.target.value);
    }
}

function toggleConnections(event) {
    const visible = event.target.checked;
    connectionLines.forEach(line => {
        line.visible = visible;
    });
}

function updateStats() {
    statsElement.textContent = `Points: ${docData.points.length} | Documents: ${Object.keys(docData.doc_map).length} | Model: ${docData.model_name}`;
}