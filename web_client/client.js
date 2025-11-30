window.onload = function () {
    console.log("Bootloader started");

    // --- Layout ---
    document.body.style.margin = '0';
    document.body.style.height = '100vh';
    document.body.style.display = 'flex';
    document.body.style.flexDirection = 'column';
    document.body.style.fontFamily = 'sans-serif';
    document.body.style.backgroundColor = '#1e1e1e';
    document.body.style.color = '#d4d4d4';

    // Status Bar
    const statusBar = document.createElement('div');
    statusBar.style.height = '25px';
    statusBar.style.backgroundColor = '#007acc';
    statusBar.style.color = 'white';
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
    mainContainer.style.borderBottom = '1px solid #333';
    document.body.appendChild(mainContainer);

    // Column 1: Property (20%)
    const colProperty = document.createElement('div');
    colProperty.style.flex = '0 0 20%';
    colProperty.style.borderRight = '1px solid #333';
    colProperty.style.overflowY = 'auto';
    colProperty.style.padding = '10px';
    mainContainer.appendChild(colProperty);

    // Column 2: Main (40%)
    const colMain = document.createElement('div');
    colMain.style.flex = '0 0 40%';
    colMain.style.borderRight = '1px solid #333';
    colMain.style.overflowY = 'auto';
    colMain.style.padding = '10px';
    colMain.innerText = 'Main Content Area';
    colMain.style.backgroundColor = '#ffe'
    mainContainer.appendChild(colMain);

    // Column 3: Detail (40%)
    const colDetail = document.createElement('div');
    colDetail.style.flex = '0 0 40%';
    colDetail.style.overflowY = 'auto';
    colDetail.style.padding = '10px';
    colDetail.innerText = 'Detail View';
    mainContainer.appendChild(colDetail);

    // Bottom Container (2 columns)
    const bottomContainer = document.createElement('div');
    bottomContainer.style.height = '200px';
    bottomContainer.style.display = 'flex';
    bottomContainer.style.borderTop = '1px solid #333';
    document.body.appendChild(bottomContainer);

    // Column 4: Repl (50%)
    const colRepl = document.createElement('div');
    colRepl.style.flex = '0 0 50%';
    colRepl.style.borderRight = '1px solid #333';
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
    replInput.style.color = '#d4d4d4';
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

    function setStatus(text, color = '#007acc') {
        statusBar.innerText = text;
        statusBar.style.backgroundColor = color;
    }

    function addLog(text, color = '#d4d4d4') {
        const entry = document.createElement('div');
        entry.innerText = text;
        entry.style.color = color;
        entry.style.borderBottom = '1px solid #333';
        entry.style.padding = '2px 0';
        colLogs.appendChild(entry);
        colLogs.scrollTop = colLogs.scrollHeight;
    }

    function addToRepl(text, color = '#d4d4d4') {
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
            th.style.borderBottom = '1px solid #555';
            th.style.padding = '5px';
            headerRow.appendChild(th);
        });
        table.appendChild(headerRow);

        models.forEach(model => {
            const row = document.createElement('tr');
            if (model.active) {
                row.style.backgroundColor = '#37373d';
            }
            if (!model.enabled) {
                row.style.color = '#888';
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
                setStatus(`Selecting model ${model.name}...`, '#ce9178');
            });

            table.appendChild(row);
        });

        colProperty.appendChild(table);
    }

    function renderSearchResults(results) {
        colMain.innerHTML = '';
        const title = document.createElement('h3');
        title.innerText = 'Search Results';
        title.style.marginTop = '0';
        colMain.appendChild(title);

        if (!results || results.length === 0) {
            colMain.innerText += 'No results found.';
            return;
        }

        results.forEach((res, index) => {
            const container = document.createElement('div');
            container.style.marginBottom = '15px';
            container.style.borderBottom = '1px solid #333';
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
                icon.style.backgroundColor = '#fff'; // Icons might have transparent bg
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
            line1.innerHTML = `<span style="color: #4ec9b0; font-weight: bold;">#${index + 1}</span> <span style="color: #ce9178;">[${res.cosine.toFixed(3)}]</span> <span style="font-weight: bold; color: #d4d4d4;">${titleText}</span>`;
            infoBlock.appendChild(line1);

            // Line 2: Authors
            if (res.metadata && res.metadata.authors && res.metadata.authors.length > 0) {
                const line2 = document.createElement('div');
                line2.style.fontSize = '0.9em';
                line2.style.color = '#888';
                line2.style.fontStyle = 'italic';
                line2.innerText = res.metadata.authors.join(', ');
                infoBlock.appendChild(line2);
            }

            // Line 3: Descriptor
            const line3 = document.createElement('div');
            line3.style.fontSize = '0.8em';
            line3.style.color = '#555';
            line3.style.fontFamily = 'monospace';
            line3.innerText = res.descriptor;
            infoBlock.appendChild(line3);

            // Text with Highlighting
            const textContainer = document.createElement('div');
            textContainer.style.marginTop = '5px';
            // textContainer.style.whiteSpace = 'pre-wrap';
            // textContainer.style.fontFamily = 'monospace';
            textContainer.style.fontSize = '12px';
            textContainer.style.lineHeight = '1.4';

            if (res.text && res.significance) {
                let currentSpan = null;
                let currentSig = -1;

                for (let i = 0; i < res.text.length; i++) {
                    const char = res.text[i];
                    const sig = res.significance[i] || 0;

                    // Optimization: Group chars with same significance
                    // Use a small epsilon for float comparison if needed, but exact match usually ok for generated arrays
                    if (sig !== currentSig) {
                        if (currentSpan) {
                            textContainer.appendChild(currentSpan);
                        }
                        currentSpan = document.createElement('span');
                        if (sig > 0) {
                            // No alpha channel needed.
                            const yellow = 255 - Math.min(sig * 255, 255);
                            currentSpan.style.backgroundColor = `rgb(255,255, ${yellow})`;
                            currentSpan.style.color = '#000'; // Black text on yellow for contrast
                        }
                        currentSig = sig;
                    }
                    currentSpan.innerText += char;
                }
                if (currentSpan) {
                    textContainer.appendChild(currentSpan);
                }
            } else {
                textContainer.innerText = res.text || '';
            }

            container.appendChild(textContainer);
            colMain.appendChild(container);
        });
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
                        addToRepl('Usage: search <query>', '#f44747');
                    }
                } else if (cmd === 'list' && args === 'models') {
                    send('list_models', '');
                } else {
                    addToRepl(`Unknown command: ${cmd}`, '#f44747');
                }
            }
        }
    });

    ws.onopen = function () {
        console.log("WebSocket connected");
        setStatus('Connected', '#007acc');

        // Request model list on startup
        send("list_models", "");
    };

    let currentProgressBar = null;

    function createProgressBar() {
        const container = document.createElement('div');
        container.style.border = '1px solid #555';
        container.style.backgroundColor = '#252526';
        container.style.height = '18px';
        container.style.position = 'relative';
        container.style.marginBottom = '2px';

        const bar = document.createElement('div');
        bar.style.backgroundColor = '#0e639c';
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
        label.style.color = '#fff';
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
                addLog(`[LOG] ${payload}`, '#9cdcfe');
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
                addLog(`[RESPONSE] ${requestCmd || ''}${final}`, '#ce9178');

                if (requestCmd === 'list_models') {
                    renderModelList(payload);
                } else if (requestCmd === 'search') {
                    renderSearchResults(payload);
                    addToRepl(`Search completed. Found ${payload ? payload.length : 0} results.`, '#4ec9b0');
                } else if (requestCmd === 'select') {
                    if (payload.status === 'ok') {
                        setStatus(`Model ${payload.selected_id} selected`, '#007acc');
                        send('list_models', ''); // Refresh list
                    } else {
                        setStatus(`Error selecting model: ${payload.error}`, '#f44747');
                    }
                }
            } else {
                addLog(`[UNKNOWN] ${JSON.stringify(data)}`, '#dcdcaa');
            }
        } catch (e) {
            console.error("Error parsing message:", e);
            addLog(`[ERROR] Raw: ${event.data}`, '#f44747');
        }
    };

    ws.onclose = function () {
        console.log("WebSocket disconnected");
        setStatus('Disconnected', '#f44747');
    };

    ws.onerror = function (error) {
        console.error("WebSocket error:", error);
        setStatus('Error', '#f44747');
    };
};
