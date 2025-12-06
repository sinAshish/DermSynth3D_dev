// Global variables
let scene, camera, renderer, controls;
let socket; // Holds the persistent WebSocket connection

const API_BASE_URL = "/api";

// --- Initialization ---
document.addEventListener("DOMContentLoaded", init);

async function init() {
    // Initialize UI components
    const meshSelect = document.getElementById("mesh-select");
    const textureSelect = document.getElementById("texture-select");
    const lesionSelect = document.getElementById("lesion-select"); // ADDED THIS LINE
    const renderBtn = document.getElementById("render-btn");
    const viewSlider = document.getElementById("view-slider");
    const viewCount = document.getElementById("view-count");

    // Set up event listeners
    meshSelect.addEventListener("change", () => updatePreview());
    textureSelect.addEventListener("change", () => updatePreview());
    lesionSelect.addEventListener("change", () => updatePreview()); // Add this line
    viewSlider.addEventListener("input", () => viewCount.textContent = viewSlider.value);
    renderBtn.addEventListener("click", handleRenderClick);

    // Establish WebSocket connection
    setupWebSocket();

    // Fetch initial config from the backend
    try {
        const response = await fetch(`${API_BASE_URL}/config`);
        if (!response.ok) {
            throw new Error(`Failed to fetch config: ${response.statusText}`);
        }
        const config = await response.json();

        // Populate dropdowns
        populateSelect(meshSelect, config.meshes);
        populateSelect(textureSelect, config.textures);

    } catch (error) {
        console.error("Error initializing app:", error);
        updateStatus("Error: Could not load initial configuration from server.", true);
    }

    // Initialize the 3D scene
    initThreeScene();

    // Load the initial preview
    updatePreview();
}

function populateSelect(selectElement, options) {
    selectElement.innerHTML = "";
    options.forEach(option => {
        const el = document.createElement("option");
        el.value = option;
        el.textContent = option;
        selectElement.appendChild(el);
    });
}


// --- UI and Status ---

function updateStatus(message, isError = false) {
    const statusElement = document.getElementById("status-message");
    statusElement.textContent = message;
    statusElement.style.color = isError ? "#ff4d4d" : "#ffc107";
}

function openTab(evt, tabName) {
    const tabcontent = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    const tablinks = document.getElementsByClassName("tab-link");
    for (let i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}


// --- 3D Preview Logic (Three.js) ---

function initThreeScene() {
    const canvasContainer = document.getElementById("preview-canvas-container");
    const canvas = document.getElementById("preview-canvas");
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x2a2a2a);
    camera = new THREE.PerspectiveCamera(50, canvasContainer.clientWidth / canvasContainer.clientHeight, 0.1, 1000);
    camera.position.set(0, 0, 3);
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    window.addEventListener('resize', () => {
        camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
    });
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
}

let currentMeshObject;
async function updatePreview() {
    const meshName = document.getElementById("mesh-select").value;
    const textureName = document.getElementById("texture-select").value;
    const numLesions = document.getElementById("lesion-select").value;

    if (!meshName) return;

    updateStatus("Loading preview...");

    if (numLesions > 0 && textureName === "No Lesion") {
        updateStatus("Warning: Cannot preview lesions with 'No Lesion' texture.", true);
        return;
    }
    if (numLesions == 0 && textureName !== "No Lesion") {
        updateStatus("Warning: Lesion textures require setting Number of Lesions > 0.", true);
        return;
    }

    try {
        const url = `${API_BASE_URL}/preview-data/${meshName}?texture_name=${textureName}&num_lesions=${numLesions}`;
        const response = await fetch(url);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `Failed to fetch preview data.`);
        }
        const data = await response.json();

        if (currentMeshObject) {
            scene.remove(currentMeshObject);
        }

        const objLoader = new THREE.OBJLoader();
        const object = objLoader.parse(data.obj_data);
        const textureLoader = new THREE.TextureLoader();
        const texture = textureLoader.load(`data:image/png;base64,${data.texture_data}`);
        
        object.traverse(function (child) {
            if (child instanceof THREE.Mesh) {
                child.material = new THREE.MeshStandardMaterial({ map: texture });
            }
        });
        
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        object.position.sub(center);
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2.0 / maxDim;
        object.scale.set(scale, scale, scale);

        currentMeshObject = object;
        scene.add(currentMeshObject);
        updateStatus("Preview loaded.", false);

    } catch (error) {
        console.error("Error updating preview:", error);
        updateStatus(`Error: ${error.message}`, true);
    }
}


// --- Rendering Logic (WebSocket) ---

function setupWebSocket() {
    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${location.host}/ws/render`;
    
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        updateStatus("Connected to render service.", false);
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.status) {
            updateStatus(`Server: ${data.status}`);
        }
        if (data.error) {
            updateStatus(`Error: ${data.error}`, true);
        }

        if (data.image) {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data.image}`;
            document.getElementById('rgb-gallery').appendChild(img);
        }
        if (data.depth) {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data.depth}`;
            document.getElementById('depth-gallery').appendChild(img);
        }
        if (data.anatomy) {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data.anatomy}`;
            document.getElementById('anatomy-gallery').appendChild(img);
        }
        if (data.segmentation) {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${data.segmentation}`;
            document.getElementById('seg-gallery').appendChild(img);
        }
    };

    socket.onerror = (error) => {
        console.error("WebSocket Error:", error);
        updateStatus("Error connecting to render service.", true);
    };

    socket.onclose = () => {
        updateStatus("Connection to render service lost. Please refresh the page to reconnect.", true);
    };
}

function handleRenderClick() {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
        updateStatus("Not connected to render service. Please wait or refresh the page.", true);
        return;
    }

    updateStatus("Initiating render... Please wait.");
    // Clear previous results
    document.getElementById('rgb-gallery').innerHTML = '';
    document.getElementById('depth-gallery').innerHTML = '';
    document.getElementById('anatomy-gallery').innerHTML = '';
    document.getElementById('seg-gallery').innerHTML = '';

    const params = {
        mesh_name: document.getElementById("mesh-select").value,
        texture_name: document.getElementById("texture-select").value,
        num_lesions: parseInt(document.getElementById("lesion-select").value, 10),
        num_views: parseInt(document.getElementById("view-slider").value, 10),
        randomize: document.getElementById("randomize-check").checked,
    };

    socket.send(JSON.stringify(params));
}
