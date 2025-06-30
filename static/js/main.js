/**
 * Main JavaScript file for Cybersecurity Drift Detection System
 */

// Global variables
let loadingModal;
let systemData = {};
let refreshInterval;
let isLoading = false;

// Initialize on DOM content loaded
document.addEventListener('DOMContentLoaded', function () {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    console.log('Initializing Cybersecurity Drift Detection System...');
    
    // Initialize Bootstrap components
    try {
        // Check if Bootstrap modal is available
        const loadingModalElement = document.getElementById('loadingModal');
        if (loadingModalElement && typeof bootstrap !== 'undefined') {
            loadingModal = new bootstrap.Modal(loadingModalElement);
        }

        // Initialize tooltips
        if (typeof bootstrap !== 'undefined') {
            var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    } catch (error) {
        console.warn('Bootstrap components initialization failed:', error);
    }

    // Check system status
    checkSystemStatus();

    // Set up periodic status checks
    refreshInterval = setInterval(checkSystemStatus, 30000); // Every 30 seconds

    // Add event listeners
    setupEventListeners();
    
    console.log('Application initialized successfully');
}

/**
 * Check system status
 */
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/system/status');
        if (response.ok) {
            const status = await response.json();
            updateSystemStatus(status);
        } else {
            console.error('Failed to fetch system status:', response.statusText);
            updateSystemStatus({ initialized: false, error: true, errorMessage: 'Failed to connect to server' });
        }
    } catch (error) {
        console.error('Error checking system status:', error);
        updateSystemStatus({ initialized: false, error: true, errorMessage: error.message });
    }
}

/**
 * Update system status indicators
 */
function updateSystemStatus(status) {
    const statusIcon = document.getElementById('systemStatus');
    const statusText = document.getElementById('systemStatusText');
    const statusCard = document.getElementById('systemStatusCard');

    // Update navbar status
    if (statusIcon && statusText) {
        if (status.initialized) {
            statusIcon.className = 'fas fa-circle text-success me-1';
            statusText.textContent = 'System Ready';
        } else if (status.error) {
            statusIcon.className = 'fas fa-circle text-danger me-1';
            statusText.textContent = 'System Error';
        } else {
            statusIcon.className = 'fas fa-circle text-warning me-1';
            statusText.textContent = 'Initializing...';
        }
    }

    // Update dashboard card if exists
    if (statusCard) {
        if (status.initialized) {
            statusCard.textContent = 'Ready';
        } else if (status.error) {
            statusCard.textContent = 'Error';
        } else {
            statusCard.textContent = 'Initializing...';
        }
    }

    // Update alert count if available
    if (status.alerts_count !== undefined) {
        updateAlertCount(status.alerts_count);
    }

    // Store status globally
    systemData.status = status;
}

/**
 * Update alert count in navigation
 */
function updateAlertCount(count) {
    const alertBadge = document.getElementById('alertCount');
    if (alertBadge) {
        alertBadge.textContent = count || 0;
        alertBadge.className = count > 0 ? 'badge bg-danger' : 'badge bg-secondary';
    }
}

/**
 * Show loading modal
 */
function showLoading(message = 'Processing request...') {
    if (isLoading) return;
    
    isLoading = true;
    
    // Update loading message if element exists
    const loadingMessage = document.querySelector('#loadingModal .modal-body p');
    if (loadingMessage) {
        loadingMessage.textContent = message;
    }
    
    if (loadingModal) {
        loadingModal.show();
    } else {
        // Fallback: create a simple loading indicator
        createFallbackLoader(message);
    }
}

/**
 * Hide loading modal
 */
function hideLoading() {
    isLoading = false;
    
    if (loadingModal) {
        loadingModal.hide();
    } else {
        // Remove fallback loader
        const fallbackLoader = document.getElementById('fallback-loader');
        if (fallbackLoader) {
            fallbackLoader.remove();
        }
    }
}

/**
 * Create fallback loader
 */
function createFallbackLoader(message) {
    const existingLoader = document.getElementById('fallback-loader');
    if (existingLoader) return;
    
    const loader = document.createElement('div');
    loader.id = 'fallback-loader';
    loader.className = 'loading-overlay';
    loader.innerHTML = `
        <div class="text-center text-white">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>${message}</p>
        </div>
    `;
    
    document.body.appendChild(loader);
}

/**
 * Show notification
 */
function showNotification(type, message, duration = 5000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 100px; right: 20px; z-index: 1050; min-width: 300px;';
    notification.innerHTML = `
        <i class="fas fa-${getNotificationIcon(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Add to document
    document.body.appendChild(notification);

    // Auto-remove after duration
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, duration);
}

/**
 * Get notification icon
 */
function getNotificationIcon(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle',
        'danger': 'times-circle'
    };
    return icons[type] || 'info-circle';
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Handle window beforeunload to clear intervals
    window.addEventListener('beforeunload', function () {
        if (refreshInterval) {
            clearInterval(refreshInterval);
        }
    });

    // Handle keyboard shortcuts
    document.addEventListener('keydown', function (e) {
        // Ctrl+R or F5 to refresh
        if ((e.ctrlKey && e.key === 'r') || e.key === 'F5') {
            e.preventDefault();
            if (typeof refreshDashboard === 'function') {
                refreshDashboard();
            } else {
                location.reload();
            }
        }

        // Escape to close modals
        if (e.key === 'Escape') {
            if (typeof bootstrap !== 'undefined') {
                const modals = document.querySelectorAll('.modal.show');
                modals.forEach(modal => {
                    const modalInstance = bootstrap.Modal.getInstance(modal);
                    if (modalInstance) {
                        modalInstance.hide();
                    }
                });
            }
        }
    });
}

/**
 * API Helper Functions
 */

/**
 * Make API request with error handling
 */
async function apiRequest(url, options = {}) {
    try {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        const response = await fetch(url, defaultOptions);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

/**
 * Get drift simulation data
 */
async function getDriftSimulation(driftType) {
    return await apiRequest('/api/drift/simulate', {
        method: 'POST',
        body: JSON.stringify({ drift_type: driftType })
    });
}

/**
 * Get visualization data
 */
async function getVisualizationData(endpoint) {
    return await apiRequest(`/api/visualizations/${endpoint}`);
}

/**
 * Acknowledge alert
 */
async function acknowledgeAlert(alertId) {
    return await apiRequest(`/api/alerts/${alertId}/acknowledge`, {
        method: 'POST'
    });
}

/**
 * Initialize system
 */
async function initializeSystem() {
    try {
        const result = await apiRequest('/api/system/initialize', { method: 'POST' });
        if (result.success) {
            showNotification('success', 'System initialized successfully');
            await checkSystemStatus();
        } else {
            throw new Error(result.message || 'Initialization failed');
        }
        return result;
    } catch (error) {
        showNotification('error', 'Failed to initialize system: ' + error.message);
        throw error;
    }
}

/**
 * Retrain model
 */
async function retrainModel() {
    try {
        const result = await apiRequest('/api/model/retrain', { method: 'POST' });
        if (result.success) {
            showNotification('success', 'Model retrained successfully');
        } else {
            throw new Error(result.error || 'Retraining failed');
        }
        return result;
    } catch (error) {
        showNotification('error', 'Failed to retrain model: ' + error.message);
        throw error;
    }
}

/**
 * Chart Helper Functions
 */

/**
 * Create empty chart
 */
function createEmptyChart(containerId, message = 'No data available') {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Chart container ${containerId} not found`);
        return;
    }

    const layout = {
        title: message,
        xaxis: { visible: false },
        yaxis: { visible: false },
        annotations: [{
            text: message,
            xref: 'paper',
            yref: 'paper',
            x: 0.5,
            y: 0.5,
            xanchor: 'center',
            yanchor: 'middle',
            showarrow: false,
            font: { size: 16, color: '#6c757d' }
        }],
        height: 400,
        margin: { t: 40, r: 40, b: 40, l: 40 }
    };

    if (typeof Plotly !== 'undefined') {
        Plotly.newPlot(containerId, [], layout);
    } else {
        container.innerHTML = `<div class="d-flex align-items-center justify-content-center h-100 text-muted">${message}</div>`;
    }
}

/**
 * Update chart with new data
 */
function updateChart(containerId, data, layout) {
    try {
        const container = document.getElementById(containerId);
        if (!container) {
            console.warn(`Chart container ${containerId} not found`);
            return;
        }

        if (typeof Plotly !== 'undefined') {
            Plotly.react(containerId, data, layout);
        } else {
            console.warn('Plotly not available for chart updates');
        }
    } catch (error) {
        console.error(`Error updating chart ${containerId}:`, error);
        createEmptyChart(containerId, 'Error loading chart');
    }
}

/**
 * Utility Functions
 */

/**
 * Format timestamp for display
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return '--';

    try {
        const date = new Date(timestamp);
        return date.toLocaleString();
    } catch (error) {
        return '--';
    }
}

/**
 * Format time ago
 */
function formatTimeAgo(timestamp) {
    if (!timestamp) return '--';

    try {
        const now = new Date();
        const time = new Date(timestamp);
        const diff = now - time;

        const seconds = Math.floor(diff / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (days > 0) return `${days}d ago`;
        if (hours > 0) return `${hours}h ago`;
        if (minutes > 0) return `${minutes}m ago`;
        return `${seconds}s ago`;
    } catch (error) {
        return '--';
    }
}

/**
 * Get severity class for Bootstrap
 */
function getSeverityClass(severity) {
    const severityMap = {
        'LOW': 'success',
        'MEDIUM': 'warning',
        'HIGH': 'danger',
        'CRITICAL': 'danger'
    };
    return severityMap[severity] || 'secondary';
}

/**
 * Get severity icon
 */
function getSeverityIcon(severity) {
    const iconMap = {
        'LOW': 'fas fa-check-circle text-success',
        'MEDIUM': 'fas fa-exclamation-triangle text-warning',
        'HIGH': 'fas fa-exclamation-circle text-danger',
        'CRITICAL': 'fas fa-times-circle text-danger'
    };
    return iconMap[severity] || 'fas fa-info-circle text-secondary';
}

/**
 * Format number with appropriate units
 */
function formatNumber(num, precision = 2) {
    if (num === null || num === undefined) return '--';

    if (num >= 1000000) {
        return (num / 1000000).toFixed(precision) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(precision) + 'K';
    } else {
        return num.toFixed(precision);
    }
}

/**
 * Validate input data
 */
function validateData(data, requiredFields = []) {
    if (!data || typeof data !== 'object') {
        return false;
    }

    return requiredFields.every(field => field in data);
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function
 */
function throttle(func, limit) {
    let inThrottle;
    return function () {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        if (navigator.clipboard) {
            await navigator.clipboard.writeText(text);
            showNotification('success', 'Copied to clipboard');
            return true;
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            showNotification('success', 'Copied to clipboard');
            return true;
        }
    } catch (error) {
        console.error('Error copying to clipboard:', error);
        showNotification('error', 'Failed to copy to clipboard');
        return false;
    }
}

/**
 * Download data as JSON file
 */
function downloadJSON(data, filename = 'data.json') {
    try {
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showNotification('success', `Downloaded ${filename}`);
    } catch (error) {
        console.error('Error downloading file:', error);
        showNotification('error', 'Failed to download file');
    }
}

/**
 * Handle errors gracefully
 */
function handleError(error, context = 'Operation') {
    console.error(`${context} failed:`, error);
    
    let message = error.message || 'An unexpected error occurred';
    if (message.includes('fetch')) {
        message = 'Unable to connect to server. Please check your connection.';
    }
    
    showNotification('error', `${context} failed: ${message}`);
}

/**
 * Export functions for global use
 */
window.CyberSecDrift = {
    // Core functions
    showLoading,
    hideLoading,
    showNotification,
    checkSystemStatus,
    initializeSystem,
    retrainModel,

    // API functions
    apiRequest,
    getDriftSimulation,
    getVisualizationData,
    acknowledgeAlert,

    // Chart functions
    createEmptyChart,
    updateChart,

    // Utility functions
    formatTimestamp,
    formatTimeAgo,
    getSeverityClass,
    getSeverityIcon,
    formatNumber,
    validateData,
    debounce,
    throttle,
    copyToClipboard,
    downloadJSON,
    handleError
};

// Export individual functions to global scope for backward compatibility
window.showLoading = showLoading;
window.hideLoading = hideLoading;
window.showNotification = showNotification;
window.getSeverityClass = getSeverityClass;
window.formatTimestamp = formatTimestamp;
window.formatTimeAgo = formatTimeAgo;
window.initializeSystem = initializeSystem;
window.retrainModel = retrainModel;
window.getDriftSimulation = getDriftSimulation;
window.createEmptyChart = createEmptyChart;
window.updateChart = updateChart;