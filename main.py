import requests
import json
import os
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from fastapi.responses import Response # Import Response for favicon

# Load environment variables (still good to keep for other keys like Tavily)
load_dotenv()

# ---
## Data Models & State
# ---

@dataclass
class HappyHourState:
    """State shared between agents"""
    user_locations: List[str] = None
    coordinates_list: List[Dict[str, float]] = None
    location_details_list: List[Dict] = None
    venues_found: List[Dict] = None
    whistle_alerts_created: List[Dict] = None
    tavily_search_results: List[Dict] = None
    search_radius_km: float = 5.0
    alert_radius_km: float = 1.0
    max_venues_per_location: int = 15
    create_whistle_alerts: bool = False
    error_messages: List[str] = None
    processing_step: str = "initialized"
    
    def __post_init__(self):
        if self.user_locations is None:
            self.user_locations = []
        if self.coordinates_list is None:
            self.coordinates_list = []
        if self.location_details_list is None:
            self.location_details_list = []
        if self.venues_found is None:
            self.venues_found = []
        if self.whistle_alerts_created is None:
            self.whistle_alerts_created = []
        if self.tavily_search_results is None:
            self.tavily_search_results = []
        if self.error_messages is None:
            self.error_messages = []

@dataclass
class VenueOffer:
    """Structured venue offer data"""
    name: str
    latitude: float
    longitude: float
    venue_type: str
    happy_hour_details: str
    distance_km: float
    offer_valid_until: str
    tags: List[str]
    location_source: str

# ---
## Utility Services
# ---

class LocationService:
    """OpenStreetMap location services"""
    
    @staticmethod
    def geocode_location(location_input: str) -> Optional[Dict]:
        """Get coordinates from location string"""
        try:
            geocoder = Nominatim(user_agent="happy_hour_finder_v2")
            location = geocoder.geocode(location_input, timeout=10)
            if not location:
                return None
            return {
                "lat": location.latitude,
                "lng": location.longitude,
                "full_address": location.address,
                "city": location.address.split(',')[0] if ',' in location.address else location_input,
                "country": location.address.split(',')[-1].strip() if ',' in location.address else "Unknown"
            }
        except Exception as e:
            print(f"Geocoding error for {location_input}: {e}")
            return None
    
    @staticmethod
    def find_venues_osm(lat: float, lng: float, radius_km: float = 5.0, max_venues: int = 15) -> List[Dict]:
        """Find venues using OpenStreetMap Overpass API"""
        radius_meters = int(radius_km * 1000)
        overpass_query = f"""
        [out:json][timeout:30];
        (
          node["amenity"~"^(bar|pub|restaurant|cafe|nightclub|biergarten)$"](around:{radius_meters},{lat},{lng});
          way["amenity"~"^(bar|pub|restaurant|cafe|nightclub|biergarten)$"](around:{radius_meters},{lat},{lng});
          node["leisure"="adult_gaming_centre"](around:{radius_meters},{lat},{lng});
          node["shop"="alcohol"](around:{radius_meters},{lat},{lng});
        );
        out center tags;
        """
        try:
            response = requests.post("https://overpass-api.de/api/interpreter", data=overpass_query, timeout=30)
            response.raise_for_status()
            data = response.json()
            venues = []
            processed_names = set()
            
            for element in data.get('elements', []):
                tags = element.get('tags', {})
                name = tags.get('name')
                if not name or name in processed_names:
                    continue
                processed_names.add(name)
                
                # Get coordinates
                if element['type'] == 'node':
                    venue_lat = element['lat']
                    venue_lng = element['lon']
                elif element['type'] == 'way' and 'center' in element:
                    venue_lat = element['center']['lat']
                    venue_lng = element['center']['lon']
                else:
                    continue
                
                # Calculate distance
                distance = geodesic((lat, lng), (venue_lat, venue_lng)).kilometers
                if distance > radius_km:
                    continue
                
                # Check amenity type
                amenity = tags.get('amenity', tags.get('leisure', tags.get('shop', 'venue')))
                if amenity not in ['bar', 'pub', 'restaurant', 'nightclub', 'biergarten', 'cafe']:
                    continue
                
                venues.append({
                    "name": name,
                    "latitude": round(venue_lat, 6),
                    "longitude": round(venue_lng, 6),
                    "venue_type": amenity,
                    "distance_km": round(distance, 2),
                    "tags": tags,
                    "source": "openstreetmap"
                })
            
            # Sort by distance and limit results
            sorted_venues = sorted(venues, key=lambda x: x['distance_km'])
            return sorted_venues[:max_venues]
        except Exception as e:
            print(f"Error fetching OSM data: {e}")
            return []

class TavilyAPIService:
    """Tavily API integration service for additional search"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = 2) -> List[Dict]:
        """Perform a search using Tavily API"""
        if not self.api_key:
            print("Tavily API key not set.")
            return []
        
        headers = {"Content-Type": "application/json"}
        data = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": max_results
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=15)
            response.raise_for_status()
            return response.json().get('results', [])
        except requests.exceptions.RequestException as e:
            print(f"Tavily search error: {e}")
            return []

class WhistleAPIService:
    """Whistle API integration service"""

    def __init__(self, api_key: str, alert_radius_km: float = 1.0):
        self.api_key = os.getenv("WHISTLE_API_KEY")
        self.alert_radius_km = alert_radius_km
        self.base_url = "http://dowhistle.herokuapp.com/v3/whistle"
        self.headers = {
            "Authorization": f"{self.api_key}",
            "Content-Type": "application/json"
        }
    
    def create_alert(self, venue_data: Dict) -> Dict:
        """Create a Whistle alert for a happy hour venue"""
        if not self.api_key or self.api_key == "":
            print("❌ Whistle API Key is missing or is the placeholder. Cannot create alert.")
            return {"error": "Whistle API Key not configured or is placeholder.", "success": False}

        try:
            # Prepare tags
            combined_tags = list(set([] + venue_data.get('tags', [])))

            # Calculate expiry for 24 hours from now
            expiry_time = datetime.now() + timedelta(hours=24)
            expiry_isoformat = expiry_time.isoformat()
            
            # Construct the payload with user-defined alert radius
            alert_payload = {
                "whistle": {
                    "provider": True,
                    "tags": combined_tags, 
                    "alertRadius": self.alert_radius_km,
                    "description": f"Happy Hour at {venue_data['name']}: {venue_data.get('happy_hour_details', 'Special offers available')}",
                    "expiry": expiry_isoformat, 
                    "latitude": venue_data['latitude'],
                    "longitude": venue_data['longitude']
                }
            }
            
            print(f"Sending alert payload to Whistle API: {json.dumps(alert_payload, indent=2)}")

            response = requests.post(self.base_url, headers=self.headers, json=alert_payload, timeout=15)
            response.raise_for_status()
            
            response_data = response.json()
            
            if response.ok:
                alert_id = response_data.get('id') or response_data.get('alertId') or response_data.get('whistleId', 'created_successfully')
                return {
                    "success": True,
                    "alert_id": alert_id,
                    "message": response_data.get('message', 'Alert created successfully'),
                    "expiry": expiry_isoformat, 
                    "created_at": datetime.now().isoformat(),
                    "alert_radius_km": self.alert_radius_km
                }
            else:
                error_message = response_data.get('message', f"HTTP status {response.status_code}")
                return {"error": error_message, "success": False, "status_code": response.status_code}

        except Exception as e:
            print(f"Error creating Whistle alert: {e}")
            return {"error": str(e), "success": False}

# ---
## Simple Sequential Workflow (No LangGraph)
# ---

class DataCollectorAgent:
    """Agent 1: Collects happy hour offers near multiple locations"""
    
    def __init__(self):
        self.location_service = LocationService()
    
    def process(self, state: HappyHourState) -> HappyHourState:
        print("🔍 DATA COLLECTOR AGENT: Starting venue search for multiple locations...")
        
        all_venues = []
        
        for location in state.user_locations:
            print(f"\n📍 Processing location: {location}")
            
            # Geocode location
            location_details = self.location_service.geocode_location(location)
            if not location_details:
                error_msg = f"Could not geocode location: {location}"
                state.error_messages.append(error_msg)
                print(f"❌ {error_msg}")
                continue
            
            state.coordinates_list.append({"latitude": location_details["lat"], "longitude": location_details["lng"]})
            state.location_details_list.append(location_details)
            print(f"✅ OSM Geocoding success for '{location}': {location_details['full_address']}")
            
            # Find venues for this location
            raw_venues = self.location_service.find_venues_osm(
                location_details["lat"], 
                location_details["lng"], 
                state.search_radius_km,
                state.max_venues_per_location
            )
            
            if not raw_venues:
                error_msg = f"No venues found near {location} within {state.search_radius_km}km"
                state.error_messages.append(error_msg)
                print(f"⚠️ {error_msg}")
                continue

            print(f"✅ OSM Venue Search success for {location}: Found {len(raw_venues)} potential venues.")
            
            # Enhance venues with happy hour data and location source
            for venue in raw_venues:
                if self._has_happy_hour_potential(venue):
                    enhanced_venue = self._enhance_venue_with_happy_hour_data(venue)
                    enhanced_venue['location_source'] = location
                    all_venues.append(enhanced_venue)
        
        state.venues_found = all_venues
        state.processing_step = "data_collection_complete"
        print(f"\n📊 DATA COLLECTOR: Total venues found across all locations: {len(all_venues)}")
        return state
    
    def _enhance_venue_with_happy_hour_data(self, venue: Dict) -> Dict:
        venue_type = venue["venue_type"]
        happy_hour_map = {
            "bar": {"offer": "Happy Hour: 25% off cocktails, wine & beer", "typical_hours": "4:00 PM - 7:00 PM", "tags": ["cocktails", "beer", "wine", "happy_hour", "bar"]},
            "pub": {"offer": "$2 off all drinks + discounted appetizers", "typical_hours": "3:00 PM - 6:00 PM", "tags": ["beer", "pub_food", "happy_hour", "appetizers"]},
            "restaurant": {"offer": "Discounted drinks with meal orders", "typical_hours": "4:30 PM - 6:30 PM", "tags": ["dining", "drinks", "happy_hour", "restaurant"]},
            "nightclub": {"offer": "Half-price drinks before 8 PM", "typical_hours": "5:00 PM - 8:00 PM", "tags": ["nightlife", "cocktails", "happy_hour", "club"]},
            "biergarten": {"offer": "$1 off all beers + pretzel specials", "typical_hours": "4:00 PM - 7:00 PM", "tags": ["beer", "outdoor", "german", "happy_hour"]},
            "cafe": {"offer": "20% off coffee & pastries", "typical_hours": "2:00 PM - 5:00 PM", "tags": ["coffee", "pastries", "afternoon", "cafe"]}
        }
        
        offer_data = happy_hour_map.get(venue_type, {
            "offer": "Happy Hour specials available", 
            "typical_hours": "varies", 
            "tags": ["happy_hour", venue_type]
        })
        
        today_end = datetime.now().replace(hour=23, minute=59, second=59)
        venue.update({
            "happy_hour_details": f"{offer_data['offer']} ({offer_data['typical_hours']})",
            "offer_valid_until": today_end.isoformat(),
            "tags": offer_data["tags"],
            "has_happy_hour": True
        })
        return venue
    
    def _has_happy_hour_potential(self, venue: Dict) -> bool:
        return venue["venue_type"] in ["bar", "pub", "restaurant", "nightclub", "biergarten", "cafe"]

class TavilySearchAgent:
    """Agent 2: Uses Tavily to find more details"""

    def __init__(self):
        self.tavily_service = TavilyAPIService(os.getenv("TAVILY_API_KEY"))

    def process(self, state: HappyHourState) -> HappyHourState:
        print("🌐 TAVILY SEARCH AGENT: Searching for additional happy hour details...")
        
        if not state.venues_found:
            print("Tavily Search: No venues found to search for.")
            state.processing_step = "tavily_search_skipped_no_venues"
            return state

        tavily_results_summary = []
        updated_venues = []
        
        for venue in state.venues_found:
            city = venue.get('location_source', '')
            query = f"{venue['name']} happy hour {city}"
            search_results = self.tavily_service.search(query, max_results=1)
            
            if search_results:
                tavily_results_summary.append({
                    "venue_name": venue['name'], 
                    "status": "enriched", 
                    "details_found": True, 
                    "source": "tavily",
                    "top_result_title": search_results[0].get('title'), 
                    "top_result_url": search_results[0].get('url')
                })
                updated_venues.append({**venue, "tavily_enriched": True})
                print(f"✅ Tavily success for '{venue['name']}': Found external details.")
            else:
                tavily_results_summary.append({
                    "venue_name": venue['name'], 
                    "status": "not_found", 
                    "details_found": False, 
                    "source": "tavily"
                })
                updated_venues.append(venue)
                print(f"ℹ Tavily for '{venue['name']}': No additional details found.")
        
        state.tavily_search_results = tavily_results_summary
        state.venues_found = updated_venues
        state.processing_step = "tavily_search_complete"
        print("🌐 TAVILY SEARCH AGENT: Completed search for venues.")
        return state

class WhistleAPICreatorAgent:
    """Agent 3: Creates Whistle API alerts"""
    
    def __init__(self, alert_radius_km: float = 1.0):
        self.whistle_service = WhistleAPIService(api_key=None, alert_radius_km=alert_radius_km)
    
    def process(self, state: HappyHourState) -> HappyHourState:
        print("📡 WHISTLE API CREATOR AGENT: Creating alerts...")
        
        if not state.create_whistle_alerts:
            print("⏩ Whistle alerts creation skipped (user opted out)")
            state.processing_step = "whistle_alerts_skipped_user_choice"
            return state
        
        if not state.venues_found:
            state.error_messages.append("No venues to create alerts for.")
            state.processing_step = "whistle_alerts_skipped_no_venues"
            return state
        
        alerts_created = []
        successful_alerts_count = 0
        
        for venue in state.venues_found:
            if venue.get("has_happy_hour", False):
                alert_result = self.whistle_service.create_alert(venue)
                if alert_result.get("success", False):
                    successful_alerts_count += 1
                    alerts_created.append({
                        "venue_name": venue["name"], 
                        "location_source": venue.get("location_source", ""),
                        "alert_id": alert_result["alert_id"], 
                        "status": "created", 
                        "expiry": alert_result["expiry"],
                        "alert_radius_km": alert_result.get("alert_radius_km", state.alert_radius_km)
                    })
                    print(f"✅ Whistle Alert success for '{venue['name']}': ID {alert_result['alert_id']}")
                else:
                    error_msg = alert_result.get("error", "Unknown error during alert creation")
                    alerts_created.append({
                        "venue_name": venue["name"], 
                        "location_source": venue.get("location_source", ""),
                        "status": "failed", 
                        "error": error_msg
                    })
                    print(f"❌ Whistle Alert failed for '{venue['name']}': {error_msg}")
            else:
                print(f"⏩ Whistle Alert skipped for '{venue['name']}': Not eligible.")
        
        state.whistle_alerts_created = alerts_created
        state.processing_step = "whistle_alerts_complete"
        print(f"📡 WHISTLE CREATOR: Successfully created {successful_alerts_count} alerts.")
        return state

# ---
## Main Application - Simple Sequential Processing
# ---

class HappyHourFinder:
    """Main application class - Simple sequential workflow"""
    
    def __init__(self):
        self.data_collector = DataCollectorAgent()
        self.tavily_searcher = TavilySearchAgent()
    
    def find_happy_hours(self, locations: List[str], radius_km: float = 5.0, 
                         max_venues: int = 15, create_alerts: bool = False, 
                         alert_radius_km: float = 1.0) -> Dict:
        """Find happy hour venues and optionally create Whistle alerts"""
        
        # Initialize whistle creator with user-defined alert radius
        self.whistle_creator = WhistleAPICreatorAgent(alert_radius_km=alert_radius_km)
        
        # Initialize state
        state = HappyHourState(
            user_locations=locations,
            search_radius_km=radius_km,
            max_venues_per_location=max_venues,
            create_whistle_alerts=create_alerts,
            alert_radius_km=alert_radius_km
        )
        
        print("🍺 HAPPY HOUR FINDER - Enhanced Sequential Implementation")
        print("=" * 70)
        print(f"📍 Searching locations: {', '.join(locations)}")
        print(f"🎯 Search radius: {radius_km}km")
        print(f"📊 Max venues per location: {max_venues}")
        print(f"📡 Create Whistle alerts: {'Yes' if create_alerts else 'No'}")
        if create_alerts:
            print(f"🔊 Alert radius: {alert_radius_km}km")
        print()
        
        try:
            # Step 1: Collect data
            state = self.data_collector.process(state)
            if not state.venues_found and not state.error_messages:
                # If no venues found and no other errors, it's a valid end state for empty results
                return self._format_results(state)
            
            # Step 2: Search with Tavily
            state = self.tavily_searcher.process(state)
            
            # Step 3: Create Whistle alerts (if requested)
            if create_alerts:
                state = self.whistle_creator.process(state)
            
            return self._format_results(state)
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            state.error_messages.append(f"Unexpected error: {str(e)}")
            return self._format_results(state)
    
    def _format_results(self, state: HappyHourState) -> Dict:
        """Format final results for API response"""
        venues_for_display = [
            v for v in state.venues_found 
            if v.get("has_happy_hour", False)
        ]
        
        overall_success = (
            len(state.error_messages) == 0 and 
            len(venues_for_display) > 0
        )

        return {
            "success": overall_success,
            "search_metadata": {
                "locations": state.user_locations, 
                "coordinates_list": state.coordinates_list,
                "search_radius_km": state.search_radius_km,
                "max_venues_per_location": state.max_venues_per_location,
                "create_whistle_alerts": state.create_whistle_alerts,
                "alert_radius_km": state.alert_radius_km,
                "timestamp": datetime.now().isoformat()
            },
            "processing_summary": {
                "locations_processed": len(state.location_details_list),
                "osm_venue_search": f"found {len(state.venues_found)} venues total" if state.venues_found else "no venues found",
                "tavily_enrichment": f"processed {len(state.tavily_search_results)} venues ({sum(1 for r in state.tavily_search_results if r.get('details_found'))} enriched)" if state.tavily_search_results else "skipped",
                "whistle_alert_creation": f"created {len([a for a in state.whistle_alerts_created if a.get('status') == 'created'])} alerts" if state.whistle_alerts_created else "no alerts created"
            },
            "results_summary": {
                "total_venues_found": len(state.venues_found), 
                "venues_eligible_for_happy_hour": len(venues_for_display),
                "whistle_alerts_created_count": len([a for a in state.whistle_alerts_created if a.get("status") == "created"]) if state.whistle_alerts_created else 0
            },
            "happy_hour_venues": venues_for_display,
            "whistle_alerts": state.whistle_alerts_created,
            "errors": state.error_messages
        }

# ---
## FastAPI Application
# ---

app = FastAPI(
    title="Happy Hour Finder API",
    description="An API to find happy hour venues and optionally create Whistle alerts.",
    version="1.0.0",
)

class HappyHourRequest(BaseModel):
    locations: List[str] = Field(..., description="List of locations (cities/addresses) to search for happy hours.")
    search_radius_km: float = Field(5.0, ge=0.1, description="Search radius in kilometers around each location.")
    max_venues_per_location: int = Field(15, ge=1, description="Maximum number of venues to return per location.")
    create_whistle_alerts: bool = Field(False, description="Whether to create Whistle alerts for found happy hour venues.")
    alert_radius_km: float = Field(1.0, ge=0.1, description="Radius in kilometers for Whistle alerts.")

# --- Added a root endpoint to avoid 404 for GET / ---
@app.get("/")
async def read_root():
    """
    Root endpoint for the Happy Hour Finder API.
    Provides a welcome message and directs to the main API endpoint.
    """
    return {
        "message": "Welcome to the Happy Hour Finder API!",
        "instructions": "Use the /find_happy_hours endpoint with a POST request to search for happy hours."
    }

# --- Added an endpoint for favicon.ico to avoid 404 logs ---
@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    """
    Returns a 204 No Content response for favicon.ico requests.
    This prevents unnecessary 404 errors in server logs from browsers.
    """
    return Response(status_code=204)

@app.post("/find_happy_hours", response_model=Dict)
async def find_happy_hours_endpoint(request: HappyHourRequest = Body(...)):
    """
    Finds happy hour venues based on the provided locations and parameters.
    Optionally creates Whistle alerts for the found venues.
    """
    try:
        finder = HappyHourFinder()
        results = finder.find_happy_hours(
            locations=request.locations,
            radius_km=request.search_radius_km,
            max_venues=request.max_venues_per_location,
            create_alerts=request.create_whistle_alerts,
            alert_radius_km=request.alert_radius_km
        )
        return {
            "status": "success" if results["success"] else "error",
            "message": "Happy Hour search complete." if results["success"] else "Happy Hour search failed.",
            "data": {
                "locations": results["search_metadata"]["locations"], 
                "coordinates_list": results["search_metadata"]["coordinates_list"],
                "venues": [
                    {
                        "name": v["name"], 
                        "location": {"lat": v["latitude"], "lng": v["longitude"]}, 
                        "distance_km": v["distance_km"],
                        "offer": v["happy_hour_details"], 
                        "venue_type": v["venue_type"], 
                        "tags": v["tags"], 
                        "valid_until": v["offer_valid_until"],
                        "location_source": v.get("location_source", "")
                    }
                    for v in results["happy_hour_venues"]
                ]
            },
            "alerts_created": results["whistle_alerts"], 
            "processing_summary": results["processing_summary"],
            "timestamp": results["search_metadata"]["timestamp"], 
            "errors": results["errors"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# ---
## Enhanced CLI Interface (kept separate)
# ---

def get_user_inputs():
    """Get all user inputs for the enhanced happy hour finder"""
    print("🍺 HAPPY HOUR FINDER - Enhanced Configuration")
    print("=" * 50)
    
    # Get cities/locations
    print("\n📍 LOCATIONS:")
    cities_input = input("Enter cities/locations (comma-separated): ").strip()
    if not cities_input:
        print("❌ Please enter at least one location")
        return None
    
    cities = [city.strip() for city in cities_input.split(',') if city.strip()]
    if not cities:
        print("❌ Please enter valid locations")
        return None
    
    # Get number of venues per location
    print("\n📊 VENUE LIMIT:")
    max_venues_str = input("How many venues per location? (default: 15): ").strip()
    max_venues = int(max_venues_str) if max_venues_str else 15
    if max_venues <= 0:
        print("❌ Max venues must be a positive number.")
        return None
    
    # Get search radius
    print("\n🎯 SEARCH RADIUS:")
    search_radius_str = input("Search radius in km? (default: 5.0): ").strip()
    search_radius = float(search_radius_str) if search_radius_str else 5.0
    if search_radius <= 0:
        print("❌ Search radius must be a positive number.")
        return None
    
    # Ask about Whistle alerts
    print("\n📡 WHISTLE ALERTS:")
    whistle_input = input("Create Whistle alerts? (y/n): ").strip().lower()
    create_alerts = whistle_input in ['y', 'yes', 'true', '1']
    
    alert_radius = 1.0
    if create_alerts:
        print("\n🔊 ALERT RADIUS:")
        try:
            alert_radius = float(input("Alert radius in km? (default: 1.0): ").strip() or "1.0")
            if alert_radius <= 0:
                print("❌ Alert radius must be a positive number, defaulting to 1.0.")
                alert_radius = 1.0
        except ValueError:
            print("❌ Invalid input for alert radius, defaulting to 1.0.")
            alert_radius = 1.0
    
    return {
        "locations": cities,
        "max_venues": max_venues,
        "search_radius": search_radius,
        "create_alerts": create_alerts,
        "alert_radius": alert_radius
    }

def display_configuration(config: Dict):
    """Display the current configuration"""
    print("\n📋 CONFIGURATION SUMMARY:")
    print("=" * 30)
    print(f"📍 Locations: {', '.join(config['locations'])}")
    print(f"📊 Max venues per location: {config['max_venues']}")
    print(f"🎯 Search radius: {config['search_radius']}km")
    print(f"📡 Create Whistle alerts: {'Yes' if config['create_alerts'] else 'No'}")
    if config['create_alerts']:
        print(f"🔊 Alert radius: {config['alert_radius']}km")
    print()

def cli_main():
    """Enhanced command line interface"""
    print("🍺 HAPPY HOUR FINDER - Enhanced Version (CLI)")
    print("Find bars/restaurants offering happy hour deals across multiple cities")
    print("=" * 70)
    
    # Get user inputs
    config = get_user_inputs()
    if not config:
        return
    
    # Display configuration
    display_configuration(config)
    
    # Confirm to proceed
    proceed = input("Proceed with search? (y/n): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("Search cancelled.")
        return
    
    print("\n🚀 Starting search...")
    print("=" * 40)
    
    # Execute search
    results = HappyHourFinder().find_happy_hours(
        locations=config["locations"],
        radius_km=config["search_radius"],
        max_venues=config["max_venues"],
        create_alerts=config["create_alerts"],
        alert_radius_km=config["alert_radius"]
    )
    
    print("\n📄 RESULTS:")
    print("=" * 40)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Save results
    filename = f"happy_hour_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {filename}")
    except Exception as e:
        print(f"❌ Could not save results: {e}")

if __name__ == "__main__":
    # To run the FastAPI application, save this code as a Python file (e.g., main.py)
    # and run 'uvicorn main:app --reload' in your terminal.
    # The CLI functionality can be called by running cli_main() directly
    # if you comment out the uvicorn command.
    
    # Example of how to run the CLI version (uncomment to use):
    # cli_main()

    # To run the FastAPI application, you typically use `uvicorn`.
    # You would run this from your terminal: uvicorn your_file_name:app --reload
    # For example, if you save this as `app.py`, run `uvicorn app:app --reload`
    print("To run the FastAPI application, save this code as a Python file (e.g., main.py)")
    print("and run 'uvicorn main:app --reload' in your terminal.")
    print("Access the API documentation at http://127.0.0.1:8000/docs after starting.")