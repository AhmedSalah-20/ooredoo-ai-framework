// ==========================================
// 1. GESTION DES ONGLETS (NAVIGATION)
// ==========================================
const pageTitles = {
    'overview': '<i class="fas fa-chart-pie"></i> Executive Overview',
    'pipeline': '<i class="fas fa-database"></i> Data Pipeline & Formats',
    'config': '<i class="fas fa-cog"></i> LoRA Config & Quantization',
    'training': '<i class="fas fa-rocket"></i> Active Training Monitor',
    'chatbot': '<i class="fas fa-comment-dots"></i> Voice Chatbot Playground',
    'benchmarking': '<i class="fas fa-balance-scale"></i> Models Benchmarking'
};

function switchTab(tabId) {
    // Cacher toutes les sections
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    // Retirer classe active des boutons
    document.querySelectorAll('.menu-btn').forEach(el => el.classList.remove('active'));
    
    // Afficher la section cible
    document.getElementById('page-' + tabId).classList.add('active');
    
    // Mettre à jour le bouton actif (basé sur le onclick)
    const activeBtn = document.querySelector(`button[onclick="switchTab('${tabId}')"]`);
    if(activeBtn) {
        activeBtn.classList.add('active');
        // Déplacer le petit point rouge "active-dot"
        const dot = document.querySelector('.active-dot');
        if(dot) dot.remove();
        activeBtn.innerHTML += '<span class="active-dot"></span>';
    }

    // Changer le titre en haut
    document.getElementById('page-title').innerHTML = pageTitles[tabId];
}

function updatePlaceholder() {
    const source = document.getElementById('dataset-source').value;
    const input = document.getElementById('dataset-path');
    
    if (source === 'huggingface') input.placeholder = "ex: linagora/linto-dataset-audio-ar-tn";
    else if (source === 'jsonl') input.placeholder = "ex: data/train.jsonl";
    else if (source === 'csv') input.placeholder = "ex: data/train.csv";
    else if (source === 'parquet') input.placeholder = "ex: data/dataset.parquet";
    else if (source === 'audio_folder') input.placeholder = "ex: data/audio_wavs/";
}

// ==========================================
// 2. SIMULATION DE L'ENTRAÎNEMENT (COURBES CHART.JS)
// ==========================================
let lossChart, valChart;
let trainingSimInterval;

// Initialiser les graphiques Chart.js
function initCharts() {
    const ctxLoss = document.getElementById('lossChart').getContext('2d');
    const ctxVal = document.getElementById('valChart').getContext('2d');

    // Chart de la Loss d'entraînement
    lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
            datasets: [{
                label: 'Training Loss',
                data: [3.5, 2.8, 2.4, 2.1, 1.8, 1.5, 1.4, 1.3, 1.25, 1.2, 1.15, 2.5],
                borderColor: '#ED1C24',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 1.2, max: 3.6, grid: { color: '#E2E8F0' } },
                x: { grid: { display: false } }
            }
        }
    });

    // Chart de la validation
    valChart = new Chart(ctxVal, {
        type: 'line',
        data: {
            labels: [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
            datasets: [
                {
                    label: 'Val Loss',
                    data: [3.9, 3.2, 2.8, 2.5, 2.3, 2.1, 1.9, 1.7, 1.6, 1.5, 1.4, 1.35],
                    borderColor: '#3B82F6',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4
                },
                {
                    label: 'Learning Rate',
                    data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // Ligne pointillée bas
                    borderColor: '#8B5CF6',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: -1.3, max: 3.9, grid: { color: '#E2E8F0' } },
                x: { grid: { display: false } }
            }
        }
    });
}

function startTrainingSim() {
    if(lossChart) lossChart.destroy();
    if(valChart) valChart.destroy();
    initCharts();

    let step = 127;
    let maxSteps = 500;
    let loss = 1.4087;

    clearInterval(trainingSimInterval);
    const consoleBox = document.getElementById('term-console');
    
    trainingSimInterval = setInterval(() => {
        step += 2;
        loss = (loss - 0.005) + (Math.random() * 0.02 - 0.01);
        if(loss < 1.1) loss = 1.1 + Math.random() * 0.1;
        
        const pct = ((step / maxSteps) * 100).toFixed(1);

        // Update UI
        if(step <= maxSteps) {
            document.getElementById('current-step').innerText = step;
            document.getElementById('current-loss').innerText = loss.toFixed(4);
            document.getElementById('chart-loss-badge').innerText = loss.toFixed(4);
            document.getElementById('prog-pct').innerText = pct + '%';
            document.getElementById('main-prog-bar').style.width = pct + '%';

            // Add console log
            const now = new Date();
            const timeStr = now.toTimeString().split(' ')[0];
            const newLog = document.createElement('div');
            newLog.className = 'highlight';
            newLog.innerText = `[${timeStr}] [STEP ${step}] loss=${loss.toFixed(4)} | grad_norm=0.790 | lr=1.87e-4`;
            
            // Retirer l'ancien highlight
            const prev = consoleBox.querySelectorAll('.highlight');
            prev.forEach(el => el.classList.remove('highlight'));
            
            consoleBox.appendChild(newLog);
            consoleBox.scrollTop = consoleBox.scrollHeight;
        } else {
            clearInterval(trainingSimInterval);
        }
    }, 2000);
}

// Initialisation au chargement
window.onload = () => {
    initCharts();
    
    // Interactions pour les tags (Modules LoRA)
    document.querySelectorAll('.tag').forEach(tag => {
        tag.addEventListener('click', () => tag.classList.toggle('active'));
    });
};

// ==========================================
// 3. RUN AUTOMATED PIPELINES (STT / LLM / TTS)
// ==========================================
async function runPipeline(event) {
    event.preventDefault();
    
    // 1. Nejbdou les valeurs mel interface
    const pipelineType = document.getElementById('pipeline-type').value;
    const datasetSource = document.getElementById('dataset-source').value;
    const datasetPath = document.getElementById('dataset-path').value;
    const statusText = document.getElementById('pipeline-status-text');
    
    if (!datasetPath) {
        alert("Please enter a dataset path!");
        return;
    }

    statusText.innerHTML = "⏳ Processing... Please wait.";
    statusText.style.color = "#3b82f6";
    
    // 2. Nbadlou l'URL mtaa l'API selon l'Pipeline choisi
    let apiUrl = "";
    if (pipelineType === "llm") {
        apiUrl = "/api/pipeline/llm/process";
    } else if (pipelineType === "stt") {
        apiUrl = "/api/pipeline/stt/process";
    } else if (pipelineType === "tts") { 
        apiUrl = "/api/pipeline/tts/process";
    }

    try {
        // 3. Nab3thou l'requête dynamique lel Backend
        const response = await fetch(apiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                source: datasetSource,
                dataset_path: datasetPath 
            })
        });

        const data = await response.json();
        
        if (data.status === "success") {
            // Mise à jour des Statistiques dans les 3 couches
            document.getElementById('stat-train-raw').innerText = data.dataset.train_raw;
            document.getElementById('stat-test-raw').innerText = data.dataset.test_raw;
            document.getElementById('stat-train-silver').innerText = data.dataset.train_silver;
            document.getElementById('stat-val-silver').innerText = data.dataset.val_silver;
            document.getElementById('stat-train-gold').innerText = data.dataset.train_gold;
            document.getElementById('stat-val-gold').innerText = data.dataset.val_gold;
            
            statusText.innerHTML = "✅ Pipeline executed successfully!";
            statusText.style.color = "#16a34a";

            // 4. Affichage dynamique mtaa les amthla (Samples)
            const samplesSection = document.getElementById('samples-section');
            const samplesContainer = document.getElementById('samples-container');
            
            if (data.samples && data.samples.length > 0) {
                samplesContainer.innerHTML = ''; 
                samplesSection.style.display = 'block'; 

                data.samples.forEach((sample, index) => {
                    // Ken el Pipeline fih Audio (STT wela TTS)
                    if (pipelineType === 'stt' || pipelineType === 'tts') {
                        const labelText = (pipelineType === 'stt') ? 'Transcript:' : 'Target Text:';
                        
                        samplesContainer.innerHTML += `
                            <div style="background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #cbd5e1; display: flex; align-items: center; gap: 20px; margin-bottom: 10px;">
                                <div style="min-width: 35px; font-weight: bold; color: #94a3b8; font-size: 18px;">#${index + 1}</div>
                                <audio controls src="${sample.audio}" style="height: 40px; width: 260px; outline: none;"></audio>
                                <div style="flex-grow: 1; font-size: 15px; color: #334155; padding-left: 15px; border-left: 2px solid #e2e8f0;">
                                    <strong style="color: #64748b; font-size: 12px; text-transform: uppercase;">${labelText}</strong><br>
                                    <span style="font-weight: 600;" dir="rtl">${sample.text}</span>
                                </div>
                            </div>
                        `;
                    } 
                    // Ken el Pipeline fih text barka (LLM)
                    else if (pipelineType === 'llm') {
                        samplesContainer.innerHTML += `
                            <div style="background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #cbd5e1; display: flex; flex-direction: column; gap: 10px; margin-bottom: 10px;">
                                <div style="font-weight: bold; color: #94a3b8; font-size: 16px;">Sample #${index + 1}</div>
                                <div style="font-size: 14px; color: #334155; background: #e2e8f0; padding: 10px; border-radius: 6px;">
                                    <strong style="color: #1e293b;">👤 User (Prompt):</strong><br>
                                    <span dir="auto">${sample.prompt}</span>
                                </div>
                                <div style="font-size: 14px; color: #15803d; background: #dcfce7; padding: 10px; border-radius: 6px;">
                                    <strong style="color: #166534;">🤖 Assistant (Response):</strong><br>
                                    <span dir="auto">${sample.response}</span>
                                </div>
                            </div>
                        `;
                    }
                });
            }

        } else {
            alert("Error: " + data.message);
            statusText.innerHTML = "❌ Pipeline failed.";
            statusText.style.color = "#dc2626";
        }
    } catch (error) {
        alert("Connection error: " + error.message);
        statusText.innerHTML = "❌ Connection failed.";
        statusText.style.color = "#dc2626";
    }

}
// ==========================================
// 4. AUTH & LOGOUT (PROTECTION DU DASHBOARD)
// ==========================================

function checkAuth() {
    const userRole = localStorage.getItem("userRole");
    if (!userRole) {
        // replace تفسخ الـ Dashboard من تاريخ المتصفح باش ما يرجعش بالـ precedent
        window.location.replace("/login"); 
    }
}

// تخدم أول ما تتحل الصفحة
document.addEventListener("DOMContentLoaded", () => {
    const userRole = localStorage.getItem("userRole");
    
    // Ken l'utilisateur mouch connecté, nraj3ouh lel login
    if (!userRole) {
        window.location.replace("/login");
        return; // نقصو التنفيذ هنا باش ما يكملش يقرا الباقي
    }

    console.log("Bienvenue ! Role actuel :", userRole);
    
    // كان السيد مدير، نظهرو الـ panel ونجيبو الـ data
    if (userRole === "Administrateur") {
        const panel = document.getElementById('admin-approval-panel');
        if (panel) {
            panel.style.display = 'block';
            fetchPendingUsers();
        }
    }
});

// تخدم حتى كي يرجع لتالي بالـ Back button (BFCache Support)
window.addEventListener('pageshow', (event) => {
    checkAuth();
});

// Fonction de Déconnexion
function logout() {
    // Fassa5 les données mel navigateur
    localStorage.removeItem("userRole");
    localStorage.removeItem("userEmail");
    // Raja3 l'utilisateur lel page login
    window.location.replace("/login");
}

// Fonction bech tjib les requêtes mel backend
async function fetchPendingUsers() {
    const res = await fetch('/api/admin/pending', {
        headers: {
            'X-User-Email': localStorage.getItem("userEmail")
        }
    });
    
    if (res.status === 403 || res.status === 401) {
        console.log("Accès refusé. Vous n'êtes pas Admin.");
        return;
    }
    
    const data = await res.json();
    const container = document.getElementById('pending-users-list');
    
    if (data.pending_users.length === 0) {
        container.innerHTML = "<p style='color: #64748b; font-size: 14px; margin: 0;'>Aucune demande en attente.</p>";
        return;
    }

    let html = '<table style="width: 100%; text-align: left; border-collapse: collapse; margin-top: 10px;">';
    data.pending_users.forEach(u => {
        html += `
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px;"><strong>${u.email}</strong></td>
                <td style="padding: 10px;"><span style="background: #e2e8f0; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">${u.role}</span></td>
                <td style="padding: 10px; text-align: right;">
                    <button onclick="handleApproval('${u.email}', 'accept')" style="background: #16a34a; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; margin-right: 5px; font-weight: bold;"><i class="fas fa-check"></i> Accepter</button>
                    <button onclick="handleApproval('${u.email}', 'reject')" style="background: #dc2626; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold;"><i class="fas fa-times"></i> Refuser</button>
                </td>
            </tr>
        `;
    });
    html += '</table>';
    container.innerHTML = html;
}

// Fonction bech tab3eth l'approbation lel backend
async function handleApproval(email, action) {
    const data = new FormData();
    data.append('email', email);
    data.append('action', action);

    const res = await fetch('/api/admin/approve', { 
        method: 'POST', 
        body: data, 
        headers: { 
            'X-User-Email': localStorage.getItem("userEmail") 
        } 
    });
    
    const result = await res.json();
    alert(result.message); // Y5arej popup feha "Compte activé"
    fetchPendingUsers(); // Ya3mel mise à jour lel liste instantanément
}