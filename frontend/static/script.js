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
                data: [3.5, 2.8, 2.4, 2.1, 1.8, 1.5, 1.4, 1.3, 1.25, 1.2, 1.15, 2.5], // Simule le pic final vu sur la capture
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