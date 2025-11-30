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
    colRepl.innerText = 'REPL Placeholder';
    bottomContainer.appendChild(colRepl);

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

            table.appendChild(row);
        });

        colProperty.appendChild(table);
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

    ws.onopen = function () {
        console.log("WebSocket connected");
        setStatus('Connected', '#007acc');

        // Request model list on startup
        send("list_models", "");
    };

    ws.onmessage = function (event) {
        try {
            const data = JSON.parse(event.data);
            const cmd = data.cmd;
            const payload = data.payload;
            const requestCmd = data.request_cmd;

            if (cmd === 'log') {
                addLog(`[LOG] ${payload}`, '#9cdcfe');
            } else if (cmd === 'response') {
                const final = data.final ? ' [FINAL]' : '';
                addLog(`[RESPONSE] ${requestCmd || ''}${final}`, '#ce9178');

                if (requestCmd === 'list_models') {
                    renderModelList(payload);
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
