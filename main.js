
let selectedFile = null;

// ============= Initialize Application =============
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    setupFileInput();
    setupDragAndDrop();
    setupSmoothScrolling();
    setupNavHighlight();
    checkSystemStatus();
}

// ============= File Input Handler =============
function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                handleFileSelect(file);
            }
        });
    }
}

// ============= Drag and Drop =============
function setupDragAndDrop() {
    const uploadBox = document.getElementById('uploadBox');
    if (!uploadBox) return;

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.add('drag-over');
    });

    uploadBox.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.remove('drag-over');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.remove('drag-over');

        const file = e.dataTransfer.files[0];
        if (file && file.type.match('image.*')) {
            handleFileSelect(file);
        } else {
            showNotification('Please upload a valid image file (PNG, JPG, or JPEG)', 'error');
        }
    });

    // Click to upload
    uploadBox.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON') {
            document.getElementById('fileInput').click();
        }
    });
}

// ============= File Selection Handler =============
function handleFileSelect(file) {
    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        showNotification('File size exceeds 16MB limit', 'error');
        return;
    }

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showNotification('Please upload PNG, JPG, or JPEG files only', 'error');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const imagePreview = document.getElementById('imagePreview');
        const uploadBox = document.getElementById('uploadBox');
        const previewSection = document.getElementById('previewSection');
        const resultsSection = document.getElementById('resultsSection');
        const fileName = document.getElementById('fileName');

        if (imagePreview) imagePreview.src = e.target.result;
        if (uploadBox) uploadBox.style.display = 'none';
        if (previewSection) previewSection.style.display = 'block';
        if (resultsSection) resultsSection.style.display = 'none';
        if (fileName) fileName.textContent = file.name;

        // Scroll to preview
        if (previewSection) {
            previewSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    };
    reader.readAsDataURL(file);
}

// ============= Reset Upload =============
function resetUpload() {
    selectedFile = null;
    
    const fileInput = document.getElementById('fileInput');
    const uploadBox = document.getElementById('uploadBox');
    const previewSection = document.getElementById('previewSection');
    const resultsSection = document.getElementById('resultsSection');
    const loader = document.getElementById('loader');

    if (fileInput) fileInput.value = '';
    if (uploadBox) uploadBox.style.display = 'block';
    if (previewSection) previewSection.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'none';
    if (loader) loader.style.display = 'none';

    // Scroll to upload
    if (uploadBox) {
        uploadBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// ============= Analyze Image =============
function analyzeImage() {
    if (!selectedFile) {
        showNotification('Please upload an image first', 'warning');
        return;
    }

    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('resultsSection');
    const previewSection = document.getElementById('previewSection');

    // Show loader
    if (loader) loader.style.display = 'block';
    if (resultsSection) resultsSection.style.display = 'none';

    // Scroll to loader
    if (loader) {
        loader.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    // Create form data
    const formData = new FormData();
    formData.append('image', selectedFile);

    // Send request with longer timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes timeout

    fetch('/predict', {
        method: 'POST',
        body: formData,
        signal: controller.signal
    })
    .then(response => {
        clearTimeout(timeoutId);
        console.log('Response received:', response.status);
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Data received:', data);
        if (loader) loader.style.display = 'none';

        if (data.error) {
            showNotification(data.error, 'error');
            return;
        }

        displayResults(data.predictions);
    })
    .catch(error => {
        clearTimeout(timeoutId);
        if (loader) loader.style.display = 'none';
        console.error('Analysis error:', error);
        console.error('Error type:', error.name);
        console.error('Error message:', error.message);
        
        if (error.name === 'AbortError') {
            showNotification('Analysis timeout. The server is taking too long to respond. Please try again.', 'error');
        } else {
            showNotification('Analysis failed. Please check if the server is running.', 'error');
        }
    });
}

// ============= Display Results =============
function displayResults(predictions) {
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsSection = document.getElementById('resultsSection');

    if (!resultsContainer || !resultsSection) return;

    resultsContainer.innerHTML = '';

    predictions.forEach((prediction, index) => {
        const card = createResultCard(prediction, index + 1);
        resultsContainer.appendChild(card);
    });

    resultsSection.style.display = 'block';

    // Scroll to results with slight delay for animation
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// ============= Create Result Card =============
function createResultCard(prediction, rank) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.style.animationDelay = `${(rank - 1) * 0.1}s`;

    // Clean disease name
    let cleanDiseaseName = prediction.disease;
    cleanDiseaseName = cleanDiseaseName.replace(/^\d+\.\s+/, '');
    cleanDiseaseName = cleanDiseaseName.replace(/\s+-\s+[\d.]+[kK]?$/, '');
    cleanDiseaseName = cleanDiseaseName.replace(/\s+\d+$/, '');

    // Determine match status
    let matchStatus, matchClass;
    if (prediction.confidence >= 80) {
        matchStatus = '✓ Highly Likely';
        matchClass = 'high';
    } else if (prediction.confidence >= 60) {
        matchStatus = '✓ Likely Match';
        matchClass = 'medium';
    } else if (prediction.confidence >= 40) {
        matchStatus = '⚠ Possible';
        matchClass = 'low';
    } else {
        matchStatus = '○ Low Confidence';
        matchClass = 'very-low';
    }

    let html = `
        <div class="result-header">
            <div>
                <div class="result-rank">Prediction #${rank}</div>
                <div class="disease-name">${cleanDiseaseName}</div>
            </div>
            <div class="confidence ${matchClass}">
                ${matchStatus}
            </div>
        </div>
    `;

    // Description
    if (prediction.description) {
        html += `
            <div class="result-section">
                <h3>📋 Description</h3>
                <p>${prediction.description}</p>
            </div>
        `;
    }

    // Symptoms
    if (prediction.symptoms && prediction.symptoms.length > 0) {
        html += `
            <div class="result-section">
                <h3>🔍 Common Symptoms</h3>
                <ul>
                    ${prediction.symptoms.map(s => `<li>${s}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    // Causes
    if (prediction.causes && prediction.causes.length > 0) {
        html += `
            <div class="result-section">
                <h3>🧬 Possible Causes</h3>
                <ul>
                    ${prediction.causes.map(c => `<li>${c}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    // Treatment
    if (prediction.treatment && prediction.treatment.length > 0) {
        html += `
            <div class="result-section">
                <h3>💊 Treatment Options</h3>
                <ul>
                    ${prediction.treatment.map(t => `<li>${t}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    // Prevention
    if (prediction.prevention && prediction.prevention.length > 0) {
        html += `
            <div class="result-section">
                <h3>🛡️ Prevention Tips</h3>
                <ul>
                    ${prediction.prevention.map(p => `<li>${p}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    // Regimens
    if (prediction.regimens && prediction.regimens.length > 0) {
        html += `
            <div class="result-section">
                <h3>📅 Recommended Regimens</h3>
                <ul>
                    ${prediction.regimens.map(r => `<li>${r}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    card.innerHTML = html;
    return card;
}

// ============= Smooth Scrolling =============
function setupSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ============= Navigation Highlight =============
function setupNavHighlight() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';
        const scrollY = window.pageYOffset;

        sections.forEach(section => {
            const sectionTop = section.offsetTop - 200;
            const sectionHeight = section.offsetHeight;

            if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

// ============= System Status Check =============
function checkSystemStatus() {
    const statusIndicator = document.getElementById('statusIndicator');
    if (!statusIndicator) return;

    fetch('/health')
        .then(response => response.json())
        .then(data => {
            const statusDot = statusIndicator.querySelector('.status-dot');
            const statusText = statusIndicator.querySelector('.status-text');

            if (data.status === 'running') {
                statusDot.style.background = '#10b981';
                statusText.textContent = data.model_loaded ? 'Model Ready' : 'System Ready';
                statusText.style.color = '#10b981';
            }
        })
        .catch(() => {
            const statusDot = statusIndicator.querySelector('.status-dot');
            const statusText = statusIndicator.querySelector('.status-text');
            
            statusDot.style.background = '#ef4444';
            statusText.textContent = 'Server Offline';
            statusText.style.color = '#ef4444';
        });
}

// ============= Notification System =============
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();

    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    const icons = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    };

    notification.innerHTML = `
        <span class="notification-icon">${icons[type] || icons.info}</span>
        <span class="notification-message">${message}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">×</button>
    `;

    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 30px;
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 20px;
        background: rgba(26, 26, 46, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(20px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        animation: slideIn 0.3s ease;
        max-width: 400px;
    `;

    // Type-specific styles
    const borderColors = {
        'info': '#667eea',
        'success': '#10b981',
        'warning': '#f59e0b',
        'error': '#ef4444'
    };
    notification.style.borderLeftWidth = '4px';
    notification.style.borderLeftColor = borderColors[type] || borderColors.info;

    // Add animation keyframes if not exists
    if (!document.querySelector('#notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            .notification-icon {
                font-size: 1.2rem;
            }
            .notification-message {
                flex: 1;
                color: #fff;
                font-size: 0.95rem;
            }
            .notification-close {
                background: none;
                border: none;
                color: rgba(255, 255, 255, 0.5);
                font-size: 1.5rem;
                cursor: pointer;
                padding: 0;
                line-height: 1;
                transition: color 0.2s;
            }
            .notification-close:hover {
                color: #fff;
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(notification);

    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

// ============= Add drag-over style =============
const dragOverStyle = document.createElement('style');
dragOverStyle.textContent = `
    .upload-box.drag-over {
        border-color: var(--primary-color) !important;
        background: rgba(102, 126, 234, 0.1) !important;
        transform: scale(1.02);
    }
    
    .result-rank {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.5);
        margin-bottom: 4px;
    }
    
    .confidence.high {
        background: rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.3);
        color: #10b981;
    }
    
    .confidence.medium {
        background: rgba(102, 126, 234, 0.1);
        border-color: rgba(102, 126, 234, 0.3);
        color: #667eea;
    }
    
    .confidence.low {
        background: rgba(245, 158, 11, 0.1);
        border-color: rgba(245, 158, 11, 0.3);
        color: #f59e0b;
    }
    
    .confidence.very-low {
        background: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.3);
        color: #ef4444;
    }
    
    .result-card {
        animation: fadeInUp 0.5s ease forwards;
        opacity: 0;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(dragOverStyle);

// ============= Periodic Status Check =============
setInterval(checkSystemStatus, 30000);
