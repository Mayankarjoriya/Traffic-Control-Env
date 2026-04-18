const MAX_CARS_VISUAL = 4;
const colorPalette = ['#e11d48', '#3b82f6', '#f59e0b', '#10b981', '#8b5cf6', '#94a3b8', '#f8fafc', '#ec4899'];

function getRandomColor() {
    return colorPalette[Math.floor(Math.random() * colorPalette.length)];
}

// Global state holding car elements to minimize DOM thrashing
const currentCars = { north: [], south: [], east: [], west: [] };

function renderCars(laneId, targetCount, ambulanceInfo) {
    const container = document.getElementById(`cars-${laneId}`);
    const displayCount = Math.min(targetCount, MAX_CARS_VISUAL);
    
    // How many do we have currently?
    const currentList = currentCars[laneId];
    
    // Add new cars if needed
    while (currentList.length < displayCount) {
        const car = document.createElement('div');
        car.className = 'car';
        
        let color = Math.random() > 0.4 ? getRandomColor() : '#cbd5e1';
        car.style.backgroundColor = color;
        
        // Setup internal divs
        const tail = document.createElement('div');
        tail.className = 'tail';
        const roof = document.createElement('div');
        roof.className = 'roof';
        car.appendChild(tail);
        car.appendChild(roof);
        
        container.appendChild(car);
        currentList.push(car);
    }
    
    // Remove extra cars if needed
    while (currentList.length > displayCount) {
        const car = currentList.pop();
        if(car && car.parentNode) {
            car.parentNode.removeChild(car);
        }
    }
    
    // Update positioning and styling for current cars
    for (let i = 0; i < currentList.length; i++) {
        const car = currentList[i];
        
        // Reset classes
        car.className = 'car';
        if (ambulanceInfo.isAmbulance && ambulanceInfo.lane === laneId && i === 0) {
            car.className = 'car ambulance';
            car.style.backgroundColor = '#f8fafc';
        }
        
        const offset = 45 + (i * 55); // Spacing between cars
        if (laneId === 'north') { 
            car.style.bottom = offset + 'px'; 
            car.style.left = 'calc(50% - 14px)'; 
            car.style.transform = 'rotate(180deg)'; 
        }
        if (laneId === 'south') { 
            car.style.top = offset + 'px'; 
            car.style.right = 'calc(50% - 14px)'; 
        }
        if (laneId === 'east') { 
            car.style.left = offset + 'px'; 
            car.style.top = 'calc(50% - 14px)'; 
            car.style.transform = 'rotate(-90deg)'; 
        }
        if (laneId === 'west') { 
            car.style.right = offset + 'px'; 
            car.style.bottom = 'calc(50% - 14px)'; 
            car.style.transform = 'rotate(90deg)'; 
        }
    }
}

async function fetchState() {
    let apiUrl = document.getElementById('api-url').value.trim();
    if (apiUrl.endsWith('/')) {
        apiUrl = apiUrl.slice(0, -1);
    }
    // Automatically detect logic if "hf.space" vs "huggingface.co/spaces/<owner>/<space>"
    // If the user inputs the spaces page, we can't fetch from it directly via API.
    // E.g. https://huggingface.co/spaces/Mayank203892/smart-traffic-grid-env
    // We try to convert it to direct space URL or just assume they gave the direct API.
    if (apiUrl.includes('huggingface.co/spaces/')) {
        const parts = apiUrl.split('huggingface.co/spaces/');
        if (parts.length > 1) {
            const pathParts = parts[1].split('/');
            if (pathParts.length >= 2) {
                const owner = pathParts[0].toLowerCase();
                const space = pathParts[1].toLowerCase();
                apiUrl = `https://${owner}-${space}.hf.space`;
            }
        }
        document.getElementById('api-url').value = apiUrl; // auto correct UI
    }

    const endpoint = `${apiUrl}/state`;

    try {
        const response = await fetch(endpoint, {
            headers: { 'Accept': 'application/json' },
            mode: 'cors'
        });
        if (!response.ok) throw new Error("HTTP error " + response.status);
        const data = await response.json();
        
        // Update Stats Deck
        document.getElementById('hud-task').innerText = `Task ${data.task_id}`;
        document.getElementById('hud-step').innerText = data.metadata?.step ?? '0';
        document.getElementById('hud-reward').innerText = data.reward.toFixed(2);
        
        const totalWait = (data.north_wait + data.south_wait + data.east_wait + data.west_wait);
        document.getElementById('hud-wait').innerText = `${totalWait}s`;

        // Update Rush Hour Alert
        const rhAlert = document.getElementById('alert-rush');
        const valRush = document.getElementById('val-rush');
        if (data.rush_hour) {
            rhAlert.classList.remove('alert-hidden');
            valRush.innerText = data.rush_hour.toUpperCase();
        } else {
            rhAlert.classList.add('alert-hidden');
            valRush.innerText = 'None';
        }

        // Update Ambulance Alert
        const ambAlert = document.getElementById('alert-ambulance');
        const valAmb = document.getElementById('val-amb');
        if (data.ambulance && data.ambulance_lane) {
            ambAlert.classList.remove('alert-hidden');
            valAmb.innerText = data.ambulance_lane.toUpperCase();
        } else {
            ambAlert.classList.add('alert-hidden');
            valAmb.innerText = 'None';
        }

        // Set Lights
        document.querySelectorAll('.traffic-light').forEach(el => el.classList.remove('green'));
        if (data.current_green_0) {
            const light = document.getElementById(`light-${data.current_green_0}`);
            if(light) light.classList.add('green');
        }
        if (data.current_green_1) {
            const light = document.getElementById(`light-${data.current_green_1}`);
            if(light) light.classList.add('green');
        }

        // Update Lane Stats Counter Text
        document.getElementById('stats-n').innerHTML = ` <span class="span-stat">${data.north_cars}</span>`;
        document.getElementById('stats-s').innerHTML = ` <span class="span-stat">${data.south_cars}</span>`;
        document.getElementById('stats-e').innerHTML = ` <span class="span-stat">${data.east_cars}</span>`;
        document.getElementById('stats-w').innerHTML = ` <span class="span-stat">${data.west_cars}</span>`;

        // Render visual cars on road
        const ambInfo = { isAmbulance: data.ambulance, lane: data.ambulance_lane };
        renderCars('north', data.north_cars, ambInfo);
        renderCars('south', data.south_cars, ambInfo);
        renderCars('east', data.east_cars, ambInfo);
        renderCars('west', data.west_cars, ambInfo);

    } catch (e) {
        console.warn("Could not fetch state:", e);
    }
}

// Initial draw then poll
fetchState();
setInterval(fetchState, 500);
