import numpy as np
import plotly.graph_objects as go
import umap
import pandas as pd
from sklearn.decomposition import PCA

import pyvista as pv
import colorsys

import json
import os
from pathlib import Path
import time


def visualize_embeddings(icotq_store):
    # 1. Collect embeddings and document mapping
    embeddings = icotq_store.embeddings_matrix.cpu().numpy()
    
    # 2. Create document mapping
    doc_mapping = {}
    doc_ids = []
    chunk_texts = []
    doc_names = []
    
    for entry in icotq_store.lib:
        if icotq_store.current_model['model_name'] in entry.get('emb_ptrs', {}):
            start, length = entry['emb_ptrs'][icotq_store.current_model['model_name']]
            doc_id = entry['desc_filename']
            for i in range(length):
                doc_ids.append(doc_id)
                chunk_texts.append(icotq_store.get_chunk(
                    entry['text'], i, 
                    icotq_store.current_model['chunk_size'], 
                    icotq_store.current_model['chunk_overlap']
                ))
                doc_names.append(entry['desc_filename'])
    
    # 3. Dimensionality reduction (PCA first for speed, then UMAP)
    if embeddings.shape[1] > 50:
        print("Reducing initial dimensions with PCA...")
        pca = PCA(n_components=50)
        embeddings_reduced = pca.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings
    
    print("Applying UMAP for final 3D projection...")
    reducer = umap.UMAP(n_components=3, metric='cosine', n_neighbors=15)
    embeddings_3d = reducer.fit_transform(embeddings_reduced)
    
    # 4. Create unique colors for each document
    unique_docs = list(set(doc_ids))
    color_map = {doc: i for i, doc in enumerate(unique_docs)}
    colors = [color_map[doc_id] for doc_id in doc_ids]
    
    # 5. Create the visualization
    df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'doc_id': doc_ids,
        'chunk_text': [text[:100] + "..." if len(text) > 100 else text for text in chunk_texts],
        'doc_name': doc_names,
        'color': colors
    })
    
    # 6. Create the Plotly figure
    fig = go.Figure(data=[go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='markers',
        marker=dict(
            size=4,
            color=df['color'],
            colorscale='rainbow',
            opacity=0.8
        ),
        text=df.apply(lambda r: f"Doc: {r['doc_name']}<br>Text: {r['chunk_text']}", axis=1),
        hoverinfo='text'
    )])
    
    # Add document connection lines (for "adhesion")
    for doc_id in unique_docs:
        doc_points = df[df['doc_id'] == doc_id]
        if len(doc_points) > 1:
            # Get centroid
            centroid = doc_points[['x', 'y', 'z']].mean()
            # Draw lines from points to centroid
            for idx, point in doc_points.iterrows():
                fig.add_trace(go.Scatter3d(
                    x=[point['x'], centroid['x']],
                    y=[point['y'], centroid['y']],
                    z=[point['z'], centroid['z']],
                    mode='lines',
                    line=dict(color=f'rgba({int(255*point["color"]/len(unique_docs))}, '
                               f'{int(100+155*point["color"]/len(unique_docs))}, 255, 0.3)'),
                    hoverinfo='none',
                    showlegend=False
                ))
    
    fig.update_layout(
        title="Document Embeddings 3D Visualization",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),
        width=1000,
        height=800,
    )
    
    fig.show()
    return fig



def visualize_embeddings_pyvista(icotq_store):
    # 1. Extract embeddings and document mapping
    embeddings = icotq_store.embeddings_matrix.cpu().numpy()
    
    # 2. Build document mapping
    doc_mapping = []
    chunk_texts = []
    doc_names = []
    doc_ids = []
    
    for entry in icotq_store.lib:
        if icotq_store.current_model['model_name'] in entry.get('emb_ptrs', {}):
            start, length = entry['emb_ptrs'][icotq_store.current_model['model_name']]
            for i in range(length):
                doc_mapping.append((start + i, entry['desc_filename'], i))
                chunk_texts.append(icotq_store.get_chunk(
                    entry['text'], i, 
                    icotq_store.current_model['chunk_size'], 
                    icotq_store.current_model['chunk_overlap']
                ))
                doc_names.append(entry['desc_filename'])
                doc_ids.append(entry['desc_filename'])
    
    # 3. Dimensionality reduction
    print("Reducing dimensions...")
    if embeddings.shape[1] > 50:
        print("Applying PCA for initial reduction...")
        pca = PCA(n_components=50)
        embeddings_reduced = pca.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings
    
    print("Applying UMAP for final 3D projection...")
    reducer = umap.UMAP(n_components=3, metric='cosine', n_neighbors=20, min_dist=0.1)
    embeddings_3d = reducer.fit_transform(embeddings_reduced)
    
    print("Dimensionality reduction completed.")
    # 4. Create document colors
    unique_docs = list(set(doc_ids))
    doc_to_id = {doc: i for i, doc in enumerate(unique_docs)}
    
    # Generate distinct colors
    colors = np.zeros((len(doc_mapping), 3))
    for i, (_, doc_id, _) in enumerate(doc_mapping):
        # Use HSV color space for better distribution
        hue = doc_to_id[doc_id] / len(unique_docs)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[i] = rgb
    
    # 5. Create PyVista point cloud
    cloud = pv.PolyData(embeddings_3d)
    cloud['doc_ids'] = doc_ids
    cloud['chunk_texts'] = chunk_texts
    cloud['colors'] = colors
    
    # 6. Create interactive plotter
    plotter = pv.Plotter()
    plotter.add_points(cloud, render_points_as_spheres=True, point_size=10, scalars='colors', rgb=True)
    
    # Add document connections (adhesion)
    for doc_id in unique_docs:
        doc_indices = [i for i, mapping in enumerate(doc_mapping) if mapping[1] == doc_id]
        if len(doc_indices) > 1:
            doc_points = embeddings_3d[doc_indices]
            centroid = np.mean(doc_points, axis=0)
            
            # Add connections as lines
            for idx in doc_indices:
                point = embeddings_3d[idx]
                line_points = np.vstack([point, centroid])
                line = pv.Line(point, centroid)
                plotter.add_mesh(line, color=colors[idx], opacity=0.3, line_width=1)
    
    # Setup callback for point picking
    def point_callback(point):
        point_id = cloud.find_closest_point(point)
        if point_id >= 0:
            doc_id = doc_ids[point_id]
            chunk_text = chunk_texts[point_id]
            plotter.add_text(
                f"Document: {doc_id}\n\nText: {chunk_text[:200]}...",
                position='upper_right',
                font_size=10,
                name='info_text'
            )
    
    plotter.enable_point_picking(callback=point_callback, show_message=False)
    plotter.show()


def prepare_embedding_visualization(icotq_store, output_dir=None, max_points=None):
    """Prepare embedding data for visualization with Three.js"""
    print("Preparing embedding visualization data...")
    
    # 1. Extract embeddings
    start_time = time.time()
    embeddings = icotq_store.embeddings_matrix.cpu().numpy()
    print(f"Extracted {embeddings.shape[0]} embeddings in {time.time() - start_time:.2f}s")
    
    if max_points and embeddings.shape[0] > max_points:
        print(f"Limiting to {max_points} points")
        # Sample points while maintaining document structure
        # (implementation would depend on specific needs)
        # For now, just take the first max_points
        embeddings = embeddings[:max_points]
    
    # 2. Build document mapping
    start_time = time.time()
    doc_mapping = []
    doc_colors = []
    doc_ids = []
    chunk_texts = []
    unique_docs = set()
    
    for entry in icotq_store.lib:
        if icotq_store.current_model['model_name'] in entry.get('emb_ptrs', {}):
            start, length = entry['emb_ptrs'][icotq_store.current_model['model_name']]
            if start >= embeddings.shape[0]:
                continue
                
            doc_id = entry['desc_filename']
            unique_docs.add(doc_id)
            
            # Process only chunks within our embedding limits
            actual_length = min(length, embeddings.shape[0] - start)
            for i in range(actual_length):
                idx = start + i
                if idx >= embeddings.shape[0]:
                    break
                    
                doc_mapping.append((idx, doc_id))
                doc_ids.append(doc_id)
                chunk_text = icotq_store.get_chunk(
                    entry['text'], i, 
                    icotq_store.current_model['chunk_size'], 
                    icotq_store.current_model['chunk_overlap']
                )
                chunk_texts.append(chunk_text[:200])  # Limit text size for performance
    
    print(f"Processed {len(doc_mapping)} document mappings in {time.time() - start_time:.2f}s")
    
    # 3. Dimensionality reduction
    start_time = time.time()
    print("Performing dimensionality reduction...")
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
        embeddings_reduced = pca.fit_transform(embeddings)
        print(f"PCA reduction completed in {time.time() - start_time:.2f}s")
        start_time = time.time()
    else:
        embeddings_reduced = embeddings
    
    reducer = umap.UMAP(n_components=3, metric='cosine', n_neighbors=15, min_dist=0.1)
    embeddings_3d = reducer.fit_transform(embeddings_reduced)
    print(f"UMAP reduction completed in {time.time() - start_time:.2f}s")
    
    # 4. Assign colors
    start_time = time.time()
    unique_doc_list = list(unique_docs)
    doc_color_map = {}
    
    for i, doc in enumerate(unique_doc_list):
        # Generate a color from HSV for better distribution
        hue = i / len(unique_doc_list)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        doc_color_map[doc] = [float(c) for c in rgb]
    
    colors = [doc_color_map[doc_id] for doc_id in doc_ids]
    print(f"Color assignment completed in {time.time() - start_time:.2f}s")
    
    # 5. Prepare output data
    start_time = time.time()
    output_data = {
        "points": embeddings_3d.tolist(),
        "colors": colors,
        "docs": doc_ids,
        "texts": chunk_texts,
        "doc_map": {doc: i for i, doc in enumerate(unique_doc_list)},
        "model_name": icotq_store.current_model['model_name']
    }
    
    # 6. Save data
    if output_dir is None:
        output_dir = os.path.join(icotq_store.root_path, "Visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"embedding_viz_{icotq_store.current_model['model_name']}.json")
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Data saved to {output_file} in {time.time() - start_time:.2f}s")
    
    # 7. Generate or copy the HTML/JS visualization files
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Embedding Visualization</title>
        <script type="importmap">
        {
          "imports": {
            "three": "https://unpkg.com/three@0.150.0/build/three.module.js",
            "three/examples/jsm/controls/OrbitControls.js": "https://unpkg.com/three@0.150.0/examples/jsm/controls/OrbitControls.js"
          }
        }
        </script>
        <style>
            body { margin: 0; overflow: hidden; }
            #info {
                position: absolute; 
                top: 10px; 
                left: 10px; 
                background: rgba(255,255,255,0.8);
                padding: 10px;
                border-radius: 5px;
                max-width: 300px;
                max-height: 300px;
                overflow: auto;
            }
            #loading {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 24px;
                background: white;
                padding: 20px;
                border-radius: 10px;
            }
        </style>
    </head>
    <body>
        <div id="info">Click a point to see document info</div>
        <div id="loading">Loading embedding data...</div>
        <script type="module">
            import * as THREE from 'three';
            import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
            
            const dataFile = './embedding_viz_{model_name}.json';
            
            // Scene setup
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf0f0f0);
            
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.z = 5;
            
            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);
            
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight.position.set(0, 1, 0);
            scene.add(directionalLight);
            
            // Raycaster for point selection
            const raycaster = new THREE.Raycaster();
            raycaster.params.Points.threshold = 0.1;
            const mouse = new THREE.Vector2();
            
            // Variables for data
            let pointsObject;
            let docData;
            let selectedPoint = null;
            let selectedPointMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });
            
            // Load data
            fetch(dataFile)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('loading').style.display = 'none';
                    docData = data;
                    createVisualization(data);
                })
                .catch(error => {
                    console.error('Error loading data:', error);
                    document.getElementById('loading').textContent = 'Error loading data';
                });
                
            function createVisualization(data) {
                // Normalize positions
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
                
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(normalizedPositions, 3));
                geometry.setAttribute('color', new THREE.Float32BufferAttribute(colorArray, 3));
                
                // Add an id attribute to help with raycasting
                const ids = new Float32Array(positions.length);
                for (let i = 0; i < positions.length; i++) {
                    ids[i] = i;
                }
                geometry.setAttribute('id', new THREE.Float32BufferAttribute(ids, 1));
                
                // Create points material
                const material = new THREE.PointsMaterial({
                    size: 0.05,
                    vertexColors: true,
                    sizeAttenuation: true,
                });
                
                // Create points object
                pointsObject = new THREE.Points(geometry, material);
                scene.add(pointsObject);
                
                // Create connection lines between points of the same document
                const docGroups = {};
                
                for (let i = 0; i < data.docs.length; i++) {
                    const docId = data.docs[i];
                    if (!docGroups[docId]) {
                        docGroups[docId] = [];
                    }
                    docGroups[docId].push(i);
                }
                
                // For each document with multiple chunks, create a centroid and connection lines
                for (const docId in docGroups) {
                    const indices = docGroups[docId];
                    if (indices.length > 1) {
                        // Calculate centroid
                        const centroid = [0, 0, 0];
                        for (const idx of indices) {
                            const pos = [
                                normalizedPositions[idx * 3],
                                normalizedPositions[idx * 3 + 1],
                                normalizedPositions[idx * 3 + 2]
                            ];
                            centroid[0] += pos[0];
                            centroid[1] += pos[1];
                            centroid[2] += pos[2];
                        }
                        centroid[0] /= indices.length;
                        centroid[1] /= indices.length;
                        centroid[2] /= indices.length;
                        
                        // Create connections with reduced opacity
                        const docColor = new THREE.Color(
                            colors[indices[0]][0],
                            colors[indices[0]][1],
                            colors[indices[0]][2]
                        );
                        
                        // Limit connections to improve performance
                        const connectionLimit = Math.min(indices.length, 20);
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
                        }
                    }
                }
                
                // Set up animation loop
                animate();
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
                // Highlight the point and all points from the same document
                const docId = docData.docs[index];
                const docIndices = [];
                
                for (let i = 0; i < docData.docs.length; i++) {
                    if (docData.docs[i] === docId) {
                        docIndices.push(i);
                    }
                }
                
                // Reset previous highlights
                if (selectedPoint) {
                    const colors = pointsObject.geometry.attributes.color;
                    for (let i = 0; i < docData.docs.length; i++) {
                        const color = docData.colors[i];
                        colors.setXYZ(i, color[0], color[1], color[2]);
                    }
                    colors.needsUpdate = true;
                }
                
                // Apply new highlights
                const colors = pointsObject.geometry.attributes.color;
                for (const i of docIndices) {
                    colors.setXYZ(i, 1.0, 1.0, 0.0); // Highlight color (yellow)
                }
                colors.needsUpdate = true;
                
                selectedPoint = index;
            }
            
            // Handle window resize
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
            
            // Add event listeners
            window.addEventListener('click', onMouseClick);
        </script>
    </body>
    </html>
    """.replace("{model_name}", icotq_store.current_model['model_name'])
    
    html_file = os.path.join(output_dir, "embedding_visualization.html")
    with open(html_file, 'w') as f:
        f.write(html_template)
    
    print(f"HTML visualization file created at {html_file}")
    print(f"Open this file in a web browser to view the visualization")
    
    return output_file, html_file
