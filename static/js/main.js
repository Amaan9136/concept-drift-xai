/**
 * Main JavaScript file for Cybersecurity Drift Detection System
 */

// Global variables
let loadingModal;
let systemData = {};
let refreshInterval;

// Initialize on DOM content loaded
document.addEventListener('DOMContentLoaded', function () {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Initialize Bootstrap components
    loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));

    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Check system status
    checkSystemStatus();

    // Set up periodic status checks
    refreshInterval = setInterval(checkSystemStatus, 30000); // Every 30 seconds

    // Add event listeners
    setupEventListeners();
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
        }
    } catch (error) {
        console.error('Error checking system status:', error);
        updateSystemStatus({ initialized: false, error: true });
    }
}

/**
 * Update system status indicators
 */
function updateSystemStatus(status) {
    const statusIcon = document.getElementById('systemStatus');
    const statusText = document.getElementById('systemStatusText');

    if (!statusIcon || !statusText) return;

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

    // Update alert count if available
    if (status.alerts_count !== undefined) {
        updateAlertCount(status.alerts_count);
    }
}

/**
 * Update alert count in navigation
 */
function updateAlertCount(count) {
    const alertBadge = document.getElementById('alertCount');
    if (alertBadge) {
        alertBadge.textContent = count;
        alertBadge.className = count > 0 ? 'badge bg-danger' : 'badge bg-secondary';
    }
}

/**
 * Show loading modal
 */
function showLoading() {
    if (loadingModal) {
        loadingModal.show();
    }
}

/**
 * Hide loading modal
 */
function hideLoading() {
    if (loadingModal) {
        loadingModal.hide();
    }
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
            }
        }

        // Escape to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const modalInstance = bootstrap.Modal.getInstance(modal);
                if (modalInstance) {
                    modalInstance.hide();
                }
            });
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
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
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
 * Chart Helper Functions
 */

/**
 * Create empty chart
 */
function createEmptyChart(containerId, message = 'No data available') {
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

    Plotly.newPlot(containerId, [], layout);
}

/**
 * Update chart with new data
 */
function updateChart(containerId, data, layout) {
    try {
        if (document.getElementById(containerId)) {
            Plotly.react(containerId, data, layout);
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

    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Format time ago
 */
function formatTimeAgo(timestamp) {
    if (!timestamp) return '--';

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
 * Local Storage Helpers
 */
function saveToLocalStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
        return true;
    } catch (error) {
        console.error('Error saving to localStorage:', error);
        return false;
    }
}

function loadFromLocalStorage(key, defaultValue = null) {
    try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
        console.error('Error loading from localStorage:', error);
        return defaultValue;
    }
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showNotification('success', 'Copied to clipboard');
        return true;
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
    saveToLocalStorage,
    loadFromLocalStorage,
    copyToClipboard,
    downloadJSON
};

// Export individual functions to global scope for backward compatibility
window.showLoading = showLoading;
window.hideLoading = hideLoading;
window.showNotification = showNotification;
window.getSeverityClass = getSeverityClass;
window.formatTimestamp = formatTimestamp;