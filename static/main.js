const MAX_CARS_VISUAL = 4;
const colorPalette = ['#e11d48', '#3b82f6', '#f59e0b', '#10b981', '#8b5cf6', '#94a3b8', '#f8fafc', '#ec4899'];

function getRandomColor() {
    return colorPalette[Math.floor(Math.random() * colorPalette.length)];
}

const currentCars = { north: [], south: [], east: [], west: [] };

function renderCars(laneId, targetCount, ambulanceInfo) {
    const container = document.getElementById(`cars-${laneId}`);
    if (!container) return;
    const displayCount = Math.min(targetCount, MAX_CARS_VISUAL);
    const currentList = currentCars[laneId];
    
    while (currentList.length < displayCount) {
        const car = document.createElement('div');
        car.className = 'car';
        car.style.backgroundColor = Math.random() > 0.4 ? getRandomColor() : '#cbd5e1';
        
        const roof = document.createElement('div');
        roof.className = 'roof';
        car.appendChild(roof);
        
        container.appendChild(car);
        currentList.push(car);
    }
    
    while (currentList.length > displayCount) {
        const car = currentList.pop();
        if(car && car.parentNode) car.parentNode.removeChild(car);
    }
    
    for (let i = 0; i < currentList.length; i++) {
        const car = currentList[i];
        car.className = 'car';
        if (ambulanceInfo.isAmbulance && ambulanceInfo.lane === laneId && i === 0) {
            car.className = 'car ambulance';
        }
        
        const offset = 40 + (i * 50); 
        if (laneId === 'north') { 
            car.style.bottom = offset + 'px'; 
            car.style.left = 'calc(50% - 14px)'; 
            car.style.transform = 'rotateZ(180deg) translateZ(5px)'; 
        } else if (laneId === 'south') { 
            car.style.top = offset + 'px'; 
            car.style.left = 'calc(50% - 14px)'; 
            car.style.transform = 'translateZ(5px)';
        } else if (laneId === 'east') { 
            car.style.left = offset + 'px'; 
            car.style.top = 'calc(50% - 14px)'; 
            car.style.transform = 'rotateZ(-90deg) translateZ(5px)'; 
        } else if (laneId === 'west') { 
            car.style.right = offset + 'px'; 
            car.style.top = 'calc(50% - 14px)'; 
            car.style.transform = 'rotateZ(90deg) translateZ(5px)'; 
        }
    }

    // V2I Flow Control
    const dataPath = document.getElementById(`path-${laneId}`);
    if (dataPath) {
        if (targetCount > 0) {
            dataPath.classList.add('active');
        } else {
            dataPath.classList.remove('active');
        }
    }
}

async function fetchState() {
    let apiUrl = document.getElementById('api-url').value.trim();
    if (apiUrl.endsWith('/')) apiUrl = apiUrl.slice(0, -1);
    
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
        document.getElementById('api-url').value = apiUrl;
    }

    const endpoint = `${apiUrl}/state`;

    try {
        const response = await fetch(endpoint, {
            headers: { 'Accept': 'application/json' },
            mode: 'cors'
        });
        if (!response.ok) throw new Error("HTTP error " + response.status);
        const data = await response.json();
        
        document.getElementById('hud-task').innerText = `Task ${data.task_id}`;
        document.getElementById('hud-step').innerText = data.metadata?.step ?? '0';
        document.getElementById('hud-reward').innerText = (data.reward || 0).toFixed(2);
        
        const totalWait = (data.north_wait + data.south_wait + data.east_wait + data.west_wait);
        document.getElementById('hud-wait').innerText = `${totalWait}s`;

        const rhAlert = document.getElementById('alert-rush');
        if (data.rush_hour) {
            rhAlert.classList.remove('alert-hidden');
            document.getElementById('val-rush').innerText = data.rush_hour.toUpperCase();
        } else {
            rhAlert.classList.add('alert-hidden');
        }

        const ambAlert = document.getElementById('alert-ambulance');
        if (data.ambulance && data.ambulance_lane) {
            ambAlert.classList.remove('alert-hidden');
            document.getElementById('val-amb').innerText = data.ambulance_lane.toUpperCase();
        } else {
            ambAlert.classList.add('alert-hidden');
        }

        document.querySelectorAll('.traffic-light').forEach(el => el.classList.remove('green'));
        if (data.current_green_0) document.getElementById(`light-${data.current_green_0}`)?.classList.add('green');
        if (data.current_green_1) document.getElementById(`light-${data.current_green_1}`)?.classList.add('green');

        document.getElementById('stats-n').innerHTML = ` <span class="span-stat">${data.north_cars}</span>`;
        document.getElementById('stats-s').innerHTML = ` <span class="span-stat">${data.south_cars}</span>`;
        document.getElementById('stats-e').innerHTML = ` <span class="span-stat">${data.east_cars}</span>`;
        document.getElementById('stats-w').innerHTML = ` <span class="span-stat">${data.west_cars}</span>`;

        const ambInfo = { isAmbulance: data.ambulance, lane: data.ambulance_lane };
        renderCars('north', data.north_cars, ambInfo);
        renderCars('south', data.south_cars, ambInfo);
        renderCars('east', data.east_cars, ambInfo);
        renderCars('west', data.west_cars, ambInfo);

    } catch (e) {
        console.warn("Fetch failed:", e);
    }
}

fetchState();
setInterval(fetchState, 500);
