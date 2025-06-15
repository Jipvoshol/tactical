import unicodedata
import os
import pickle
import datetime
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import base64
from typing import Dict, Tuple, List, Optional, Any
from sklearn.pipeline import Pipeline
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import requests 
import time      
import traceback 


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*mean of empty slice.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting behavior in replace is deprecated.*")


# --- Configuratie & Globale Variabelen ---
DEFAULT_TICKETS_PATH = "tickets_processed.csv"
DEFAULT_ARTISTS_PATH = "artist_data.csv"  # Fallback artiesten data (beheerd door notebook)
DEFAULT_LINEUP_PATH = "line_up_processed.csv" # Minder kritisch voor live tool, meer voor training/analyse
DEFAULT_MODELS_DIR = "." # Modellen worden verwacht in dezelfde map als het script

# Omgevingsvariabelen voor API-sleutels
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "19b91c20d3f14fe896aa6019a45afb3a")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "34f866baa3f64224a395d7136a9aacde")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_KEY") or "a1b0ba0a6a056cead5571827d977c8bb"

# Spotify API Endpoints
SPOTIFY_API_URL_TOKEN = "https://accounts.spotify.com/api/token" 
SPOTIFY_API_URL_V1 = "https://api.spotify.com/v1" 
_spotify_token_cache = {"token": None, "expires_at": 0.0}

OPENWEATHER_API_URL_FORECAST = "https://api.openweathermap.org/data/2.5/forecast"

best_models: Dict[int, Tuple[Pipeline, bool]] = {} # Globale dictionary voor geladen modellen

# ===============================================================
# Algemene Helper Functies
# ===============================================================
def debug_print(message: str, verbose: bool = True) -> None:
    """Helper om debug berichten te printen als verbose True is."""
    if verbose:
        print(f"[DEBUG] {str(message)}")

def normalize_text(text: Any) -> str:
    """Normaliseert tekst: lowercase, strip, verwijder accenten."""
    if pd.isna(text) or text is None: 
        return ""
    if not isinstance(text, str):
        text_str = str(text) # Converteer naar string als het geen string is
    else:
        text_str = text
    
    try:
        nfkd_form = unicodedata.normalize('NFKD', text_str)
        ascii_text = nfkd_form.encode('ASCII', 'ignore').decode('utf-8')
        return ascii_text.lower().strip()
    except Exception as e:
        debug_print(f"Fout bij normaliseren tekst '{text_str}': {e}. Gebruik fallback normalisatie.", True)
        # Fallback voor onverwachte encoding issues
        return text_str.lower().strip()

def get_real_cities_present_in_data(tickets: pd.DataFrame, verbose: bool = False) -> List[str]:
    """Retourneert een lijst met unieke, genormaliseerde steden uit de ticketdata."""
    if 'city' not in tickets.columns or tickets['city'].empty:
        debug_print("Kolom 'city' niet gevonden of leeg in ticketdata voor get_real_cities_present_in_data.", verbose)
        return []
    try:
        # Probeer geonamescache te gebruiken voor validatie
        import geonamescache
        gc = geonamescache.GeonamesCache()
        all_geo_cities_normalized = {normalize_text(city_info['name']) for city_info in gc.get_cities().values()}
        
        data_cities_normalized = set(tickets['city'].dropna().astype(str).apply(normalize_text).unique())
        
        # Steden die zowel in data als (genormaliseerd) in geonamescache voorkomen
        real_cities = [city for city in data_cities_normalized if city in all_geo_cities_normalized]
        
        if not real_cities and data_cities_normalized:
             debug_print("Geen directe match met geonamescache steden, gebruik alle genormaliseerde steden uit input data.", verbose)
             real_cities = list(data_cities_normalized)
        elif not data_cities_normalized:
             debug_print("Geen steden in input data om te verwerken.", verbose)
             return []


        debug_print(f"{len(real_cities)} unieke steden geïdentificeerd voor 'known_cities'.", verbose)
        return real_cities
    except ImportError:
        debug_print("Module 'geonamescache' niet geïnstalleerd. Gebruik unieke steden direct uit data.", verbose)
        return list(tickets['city'].dropna().astype(str).apply(normalize_text).unique())
    except Exception as e:
        debug_print(f"Algemene fout bij ophalen/valideren steden: {e}. Gebruik unieke steden uit data.", verbose)
        return list(tickets['city'].dropna().astype(str).apply(normalize_text).unique())

def get_closest_trained_model_T(requested_T: int) -> int:
    """Vindt de T-waarde van het dichtstbijzijnde beschikbare getrainde model."""
    global best_models
    available_Ts = sorted(list(best_models.keys()))
    if not available_Ts:
        raise ValueError("Geen modellen beschikbaar (best_models is leeg). Roep find_and_load_models() eerst aan.")
    
    # Behandel negatieve T (kan gebeuren als event in het verleden ligt t.o.v. 'today')
    if requested_T < 0:
        actual_requested_T = 0 # Zet om naar 0 voor model selectie
        debug_print(f"Opgevraagde T ({requested_T}) is negatief, behandeld als T=0 voor model selectie.", True)
    else:
        actual_requested_T = requested_T

    if actual_requested_T in available_Ts:
        return actual_requested_T
    
    if not available_Ts: # Dubbelcheck voor het geval best_models leeg is
        raise ValueError("Kan dichtstbijzijnde T niet vinden: geen modellen geladen.")

    # Als requested_T kleiner is dan de kleinste beschikbare T, gebruik de kleinste.
    if actual_requested_T < available_Ts[0]:
        closest_T = available_Ts[0]
        print(f"INFO: Opgevraagde T={actual_requested_T} is kleiner dan kleinste model (T={closest_T}). Gebruikt T={closest_T}.")
        return closest_T
        
    # Als requested_T groter is dan de grootste beschikbare T, gebruik de grootste.
    if actual_requested_T > available_Ts[-1]:
        closest_T = available_Ts[-1]
        print(f"INFO: Opgevraagde T={actual_requested_T} is groter dan grootste model (T={closest_T}). Gebruikt T={closest_T}.")
        return closest_T

    # Vind anders de dichtstbijzijnde T (kan zowel kleiner als groter zijn)
    closest_T = min(available_Ts, key=lambda x: abs(x - actual_requested_T))
    print(f"INFO: Geen model voor T={actual_requested_T}. Gebruikt dichtstbijzijnde model T={closest_T}.")
    return closest_T

def find_artist_column(df: pd.DataFrame, common_names: Optional[List[str]] = None) -> Optional[str]:
    """Zoekt naar een kolom die waarschijnlijk artiestennamen bevat in de DataFrame."""
    if df is None or df.empty:
        return None
    if common_names is None:
        common_names = ['artist', 'artist_name', 'artiest', 'artiest_naam', 'name']
    
    normalized_df_cols = {col.lower().strip(): col for col in df.columns}

    for common_name in common_names:
        if common_name in normalized_df_cols:
            return normalized_df_cols[common_name] # Return originele kolomnaam
            
    # Fallback: zoek naar kolommen die een van de namen bevatten
    for common_name_part in common_names:
        for lower_col, original_col in normalized_df_cols.items():
            if common_name_part in lower_col:
                return original_col # Return originele kolomnaam
    return None

# ===============================================================
# Spotify API Functies
# ===============================================================
def get_spotify_access_token(verbose: bool = True) -> Optional[str]:
    """Haalt een Spotify Access Token op of gebruikt een gecachte token."""
    global _spotify_token_cache
    current_time = time.time()

    if _spotify_token_cache.get("token") and _spotify_token_cache.get("expires_at", 0) > current_time + 60: # 60s buffer
        return _spotify_token_cache["token"]


    try:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "client_credentials"}
        
        response = requests.post(
            SPOTIFY_API_URL_TOKEN, 
            headers=headers, 
            data=data, 
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET), 
            timeout=10
        )
        response.raise_for_status() 
        
        token_data = response.json()
        access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in", 3600) # Default op 1 uur

        if not access_token:
            debug_print("FOUT: 'access_token' niet gevonden in Spotify token response.", verbose)
            return None

        _spotify_token_cache["token"] = access_token
        _spotify_token_cache["expires_at"] = current_time + expires_in
        # debug_print("Nieuwe Spotify token succesvol verkregen.", verbose)
        return access_token
    except requests.exceptions.Timeout:
        debug_print(f"FOUT: Timeout bij aanvragen Spotify token naar {SPOTIFY_API_URL_TOKEN}", verbose)
    except requests.exceptions.HTTPError as http_err:
        debug_print(f"FOUT (HTTPError) bij aanvragen Spotify token: {http_err} - Response: {http_err.response.text}", verbose)
    except requests.exceptions.RequestException as req_err:
        debug_print(f"FOUT (RequestException) bij aanvragen Spotify token: {req_err}", verbose)
    except KeyError as key_err:
        debug_print(f"FOUT: Veld {key_err} niet gevonden in Spotify token response. Response: {token_data if 'token_data' in locals() else 'Onbekend'}", verbose)
    except Exception as e: 
        debug_print(f"ALGEMENE FOUT bij ophalen Spotify token: {e}", verbose)
        debug_print(traceback.format_exc(), verbose)
    return None

def search_spotify_artist_data(artist_name_query: str, token: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """
    Zoekt een artiest op Spotify op naam (genormaliseerd) en retourneert dictionary met geselecteerde metrics.
    """
    if not token:
        debug_print(f"Geen Spotify token beschikbaar voor zoeken naar '{artist_name_query}'.", verbose)
        return None
    
    headers = {"Authorization": f"Bearer {token}"}
    # De artist_name_query wordt hier al genormaliseerd verwacht, maar een extra normalize_text kan geen kwaad.
    normalized_query = normalize_text(artist_name_query)
    params = {"q": normalized_query, "type": "artist", "limit": 1} 

    try:
        response = requests.get(f"{SPOTIFY_API_URL_V1}/search", headers=headers, params=params, timeout=10)
        response.raise_for_status()
        search_results = response.json()
        
        items = search_results.get("artists", {}).get("items", [])
        if items:
            artist_data_raw = items[0] 
            formatted_data = {
                'artist_api_name': artist_data_raw.get('name'), # Originele naam van API voor eventuele check
                'SpotifyPopularity': artist_data_raw.get('popularity'),
                'SpotifyFollowers': artist_data_raw.get('followers', {}).get('total'),
                'genres_list': artist_data_raw.get('genres', []), 
                'spotify_id': artist_data_raw.get('id'),
                'spotify_url': artist_data_raw.get('external_urls', {}).get('spotify')
            }
            return formatted_data
        else:
            # debug_print(f"Artiest '{normalized_query}' niet gevonden op Spotify.", verbose) # Kan veel output geven
            return None
    except requests.exceptions.Timeout:
        debug_print(f"FOUT: Timeout bij zoeken artiest '{normalized_query}' op Spotify.", verbose)
    except requests.exceptions.HTTPError as http_err:
        debug_print(f"FOUT (HTTPError) bij zoeken artiest '{normalized_query}' op Spotify: {http_err} - Response: {http_err.response.text}", verbose)
    except requests.exceptions.RequestException as req_err:
        debug_print(f"FOUT (RequestException) bij zoeken artiest '{normalized_query}' op Spotify: {req_err}", verbose)
    except Exception as e: 
        debug_print(f"ALGEMENE FOUT bij verwerken Spotify data voor '{normalized_query}': {e}", verbose)
        debug_print(traceback.format_exc(), verbose)
    return None

# ===============================================================
# OpenWeatherMap API Functions
# ===============================================================
def get_weather_forecast_for_event(
    event_date: pd.Timestamp, 
    location: Optional[str], 
    verbose: bool = True
) -> Optional[Dict[str, float]]:
    """
    Get weather forecast using OpenWeatherMap API, following the notebook logic.
    - Gets 5-day forecast in 3-hour intervals
    - Aggregates to daily values (t_max, t_min, rain_fall, max_wind)
    - If event_date is within forecast range, uses specific values for that day
    - Otherwise, uses average values across the entire forecast
    
    Returns a dictionary with weather features or None if API fails.
    """
    if not OPENWEATHER_API_KEY:
        debug_print("OpenWeatherMap API Key niet (correct) geconfigureerd. Kan live weer niet ophalen.", verbose)
        return None
        
    if not location or pd.isna(location):
        debug_print(f"Geen geldige locatie opgegeven voor weer. Amsterdam wordt gebruikt als fallback.", verbose)
        location = "Amsterdam"  # Fallback locatie

    normalized_location = normalize_text(location)
    debug_print(f"Weather forecast ophalen voor {normalized_location} rond {event_date.date()}...", verbose)
    
    params = {"q": normalized_location, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    
    try:
        response = requests.get(OPENWEATHER_API_URL_FORECAST, params=params, timeout=15)
        response.raise_for_status()
        forecast_data = response.json()
    except requests.exceptions.Timeout:
        debug_print(f"FOUT: Timeout bij ophalen weerdata voor {normalized_location}.", verbose)
        return None
    except requests.exceptions.HTTPError as http_err:
        debug_print(f"FOUT: HTTP error bij ophalen weerdata voor {normalized_location}: {http_err}. Response: {http_err.response.text if hasattr(http_err, 'response') else 'Geen response'}", verbose)
        return None
    except requests.exceptions.RequestException as req_err:
        debug_print(f"FOUT: Request error bij ophalen weerdata voor {normalized_location}: {req_err}", verbose)
        return None
    except Exception as e_gen_weather: 
        debug_print(f"ALGEMENE FOUT bij ophalen weerdata: {e_gen_weather}", verbose)
        return None

    entries = forecast_data.get("list", [])
    if not entries:
        debug_print(f"INFO: Geen 3-uurlijkse forecast data ontvangen van API voor {normalized_location}.", verbose)
        return None

    processed_entries = []
    for entry in entries:
        try:
            ts = datetime.datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S")
            processed_entries.append({
                'timestamp': ts,
                'date_obj': ts.date(),
                't_max': entry['main']['temp_max'],
                't_min': entry['main']['temp_min'],
                'rain_fall': entry.get("rain", {}).get("3h", 0.0),
                'max_wind': entry.get("wind", {}).get("gust", entry.get("wind", {}).get("speed", 0.0)) * 3.6
            })
        except KeyError as e:
            debug_print(f"WAARSCHUWING: Key niet gevonden in API entry: {e}. Entry: {entry}", verbose)
            continue
        except Exception as e_proc:
            debug_print(f"FOUT: Kon entry niet verwerken: {e_proc}. Entry: {entry}", verbose)
            continue
            
    if not processed_entries:
        debug_print(f"INFO: Geen verwerkbare 3-uurlijkse forecast data na parsing voor {normalized_location}.", verbose)
        return None

    forecast_df = pd.DataFrame(processed_entries)
    
    # Groepeer per datum en aggregeer volgens de notebook logica
    daily_aggregated_weather_df = forecast_df.groupby('date_obj').agg(
        t_max=('t_max', 'max'),
        t_min=('t_min', 'min'),
        rain_fall=('rain_fall', 'sum'),
        max_wind=('max_wind', 'max')
    ).reset_index()

    if daily_aggregated_weather_df.empty:
        debug_print(f"INFO: Geen dagelijkse geaggregeerde weerdata beschikbaar voor {normalized_location}.", verbose)
        return None

    target_date_obj = event_date.normalize().date()

    # Probeer specifieke dag in forecast te vinden
    specific_day_weather = daily_aggregated_weather_df[daily_aggregated_weather_df['date_obj'] == target_date_obj]

    if not specific_day_weather.empty:
        row = specific_day_weather.iloc[0]
        debug_print(f"Specifieke weervoorspelling gevonden voor {target_date_obj} in {normalized_location}.", verbose)
        return {
            't_max': row['t_max'],
            't_min': row['t_min'],
            'rain_fall': row['rain_fall'],
            'max_wind': row['max_wind']
        }
    else:
        # Fallback naar 5-daags gemiddelde als specifieke dag niet in de voorspelling zit (bijv. >5 dagen vooruit)
        debug_print(f"INFO: Geen specifieke forecast voor {target_date_obj} in {normalized_location}. 5-daags gemiddelde wordt gebruikt.", verbose)
        return {
            't_max': daily_aggregated_weather_df['t_max'].mean(),
            't_min': daily_aggregated_weather_df['t_min'].mean(),
            'rain_fall': daily_aggregated_weather_df['rain_fall'].mean(),
            'max_wind': daily_aggregated_weather_df['max_wind'].mean()
        }

def fetch_live_weather_for_event_day(
    event_date: pd.Timestamp, 
    location: Optional[str], 
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Haalt de weersvoorspelling op voor de specifieke event_date.
    
    Retourneert een dictionary met {'t_max': ..., 't_min': ..., 'rain_fall': ..., 'max_wind': ...}
    of None.
    """
    return get_weather_forecast_for_event(event_date, location, verbose)

# ===============================================================
# Functie voor ophalen Live Artiestendata (Spotify + Fallback)
# ===============================================================
def fetch_live_artist_data_for_lineup(
    artist_names_list: List[str], 
    artists_df_fallback: pd.DataFrame, 
    verbose: bool = True
) -> pd.DataFrame:
    """
    Haalt live artiestendata op via de Spotify API voor de opgegeven artiestennamen.
    Gebruikt artists_df_fallback (geladen uit DEFAULT_ARTISTS_PATH) als een artiest
    niet live gevonden kan worden of als de API call faalt.
    Focust op het vullen van SpotifyPopularity en SpotifyFollowers via de API.

    Args:
        artist_names_list (List[str]): Lijst met (niet-genormaliseerde) artiestennamen.
        artists_df_fallback (pd.DataFrame): DataFrame met fallback artiestendata.
        Verwacht kolom 'artist' (genormaliseerd) en 'Date'.
        verbose (bool): Indien True, print debug informatie.

    Returns:
        pd.DataFrame: Een DataFrame met artiestenmetrics, klaar voor engineer_features.
                      Bevat minimaal 'artist' en 'Date'. 
    """
    collected_artist_data_list = [] # Lijst om dictionaries per artiest op te slaan
    spotify_access_token = get_spotify_access_token(verbose=verbose)

    # Definieer alle kolomnamen die we verwachten in de uiteindelijke artists_df
    # die naar engineer_features gaat. Dit zorgt voor een consistente structuur.
    expected_columns_for_features = [
        'artist', 'Date', 
        'SpotifyPopularity', 'SpotifyFollowers',
        'ChartmetricScore', 'TikTokViews', 
        'genres_list', 'spotify_id', 'spotify_url'      # Extra's van Spotify
    ]

    for original_artist_name_in_lineup in artist_names_list:
        normalized_artist_name_query = normalize_text(original_artist_name_in_lineup)
        
        # Sla over als de genormaliseerde naam leeg is (bijv. na het strippen van speciale tekens)
        if not normalized_artist_name_query:
            debug_print(f"Overgeslagen: lege artiestennaam na normalisatie van '{original_artist_name_in_lineup}'.", verbose)
            continue

        # Initialiseer een dictionary voor de data van deze artiest met NaNs voor alle verwachte kolommen
        current_artist_entry_dict = {col_name: np.nan for col_name in expected_columns_for_features}
        current_artist_entry_dict['artist'] = normalized_artist_name_query # Gebruik de genormaliseerde naam als key
        current_artist_entry_dict['Date'] = pd.Timestamp.now(tz='UTC').normalize() # Standaard datum is vandaag (voor live data)

        # 1. Probeer live Spotify data op te halen
        live_spotify_data_dict = None
        if spotify_access_token:
            live_spotify_data_dict = search_spotify_artist_data(normalized_artist_name_query, spotify_access_token, verbose=verbose)
        
        if live_spotify_data_dict:
            debug_print(f"Live Spotify data succesvol opgehaald voor '{normalized_artist_name_query}'.", verbose)
            # Update de current_artist_entry_dict met de live opgehaalde Spotify waarden
            # De 'artist_api_name' wordt gebruikt voor logging/vergelijking, maar 'artist' blijft de genormaliseerde query.
            for key_from_api, value_from_api in live_spotify_data_dict.items():
                if key_from_api in current_artist_entry_dict: # Alleen updaten als de kolom verwacht wordt
                    current_artist_entry_dict[key_from_api] = value_from_api
            # De 'Date' van live_spotify_data_dict (vandaag) wordt gebruikt.
        

        collected_artist_data_list.append(current_artist_entry_dict)

    # Maak de uiteindelijke DataFrame
    final_artists_df = pd.DataFrame(collected_artist_data_list)
    
    # Zorg ervoor dat alle verwachte kolommen bestaan, zelfs als ze alleen NaNs bevatten
    for col_final_check in expected_columns_for_features:
        if col_final_check not in final_artists_df.columns:
            final_artists_df[col_final_check] = np.nan # Voeg kolom toe met NaNs
            
    # Converteer 'Date' kolom expliciet naar datetime64[ns, UTC] voor consistentie
    if 'Date' in final_artists_df.columns:
        final_artists_df['Date'] = pd.to_datetime(final_artists_df['Date'], errors='coerce', utc=True)
    else: # Mocht 'Date' er om een of andere reden niet zijn
        final_artists_df['Date'] = pd.NaT

    return final_artists_df.reindex(columns=expected_columns_for_features) # Zorgt voor juiste kolomvolgorde en aanwezigheid

# ===============================================================
# Model Laden Functie
# ===============================================================
def find_and_load_models(models_dir: str = DEFAULT_MODELS_DIR, verbose: bool = True) -> bool:
    """
    Zoekt naar en laadt alle beschikbare ticketvoorspellingsmodellen (.pkl bestanden).
    Vult de globale 'best_models' dictionary.
    Retourneert True als tenminste één model succesvol is geladen, anders False.
    """
    global best_models
    best_models.clear() # Maak leeg voor het geval de functie opnieuw wordt aangeroepen

    if not os.path.isdir(models_dir):
        debug_print(f"FOUT: Modellen directory '{models_dir}' niet gevonden.", verbose)
        return False

    model_files_found = []
    for filename in os.listdir(models_dir):
        if filename.startswith("ticket_prediction_model_T") and filename.endswith(".pkl"):
            model_files_found.append(os.path.join(models_dir, filename))
    
    if not model_files_found:
        debug_print(f"WAARSCHUWING: Geen modelbestanden (ticket_prediction_model_T*.pkl) gevonden in '{models_dir}'.", verbose)
        return False
    
    models_loaded_count = 0
    for model_file_path in model_files_found:
        try:
            # Haal T-waarde uit bestandsnaam, bijv. ticket_prediction_model_T43_CV_XGBRegressor.pkl
            # Dit deel is afhankelijk van de exacte naamgevingsconventie van je modelbestanden.
            base_filename = os.path.basename(model_file_path)
            t_value_str = base_filename.split('_T')[1].split('_')[0]
            t_value_int = int(t_value_str)
            
            with open(model_file_path, 'rb') as f:
                loaded_model_object = pickle.load(f)
            
            
            # Verwacht formaat: een tuple (pipeline_object, target_transformed_boolean)
            if isinstance(loaded_model_object, tuple) and \
               len(loaded_model_object) == 2 and \
               isinstance(loaded_model_object[0], Pipeline) and \
               isinstance(loaded_model_object[1], bool):
                best_models[t_value_int] = loaded_model_object
                models_loaded_count += 1
                # debug_print(f"Model voor T={t_value_int} succesvol geladen van: {model_file_path}", verbose)
            else:
                # Ondersteun ook het oudere formaat waarbij het model in een dictionary zat
                # { T_value: (pipeline, bool) }
                if isinstance(loaded_model_object, dict) and loaded_model_object:
                    # Neem de eerste (en vermoedelijk enige) entry uit de dictionary
                    # De T-waarde uit de bestandsnaam is leidend.
                    _, model_tuple_from_dict = next(iter(loaded_model_object.items()))
                    if isinstance(model_tuple_from_dict, tuple) and \
                       len(model_tuple_from_dict) == 2 and \
                       isinstance(model_tuple_from_dict[0], Pipeline) and \
                       isinstance(model_tuple_from_dict[1], bool):
                        best_models[t_value_int] = model_tuple_from_dict
                        models_loaded_count += 1
                        # debug_print(f"Model voor T={t_value_int} (uit dict) succesvol geladen van: {model_file_path}", verbose)
                    else:
                        debug_print(f"WAARSCHUWING: Modelbestand {model_file_path} (dict formaat) bevat onverwachte structuur.", verbose)    
                else:
                    debug_print(f"WAARSCHUWING: Modelbestand {model_file_path} heeft een onverwacht formaat en is overgeslagen.", verbose)
        except ValueError:
            debug_print(f"FOUT: Kon T-waarde niet parsen uit bestandsnaam: {model_file_path}", verbose)
        except Exception as e:
            debug_print(f"Algemene FOUT bij laden van model {model_file_path}: {e}", verbose)
            debug_print(traceback.format_exc(), verbose)
            
    if models_loaded_count > 0:
        debug_print(f"Totaal {models_loaded_count} modellen succesvol geladen voor T-waarden: {sorted(best_models.keys())}", True) # Altijd printen
        return True
    else:
        debug_print("WAARSCHUWING: Geen modellen succesvol geladen na doorlopen van bestanden.", True) # Altijd printen
        return False

# ===============================================================
# Data Laden Functie (Vereenvoudigd voor Live Tool)
# ===============================================================
def load_data(
    tickets_path: str = DEFAULT_TICKETS_PATH, 
    artists_fallback_path: str = DEFAULT_ARTISTS_PATH,
    lineup_path_historical: Optional[str] = DEFAULT_LINEUP_PATH 
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Laadt de basis datasets: 
    - tickets_df van tickets_path.
    - artists_df_fallback van artists_fallback_path.
    - lineup_df_historical (optioneel) van lineup_path_historical.
    Normaliseert relevante kolommen ('event_name', 'artist', 'city').
    """
    debug_print(f"Start laden basisdata...", True) # Altijd printen

    # 1. Tickets Data
    tickets_df = pd.DataFrame()
    try:
        debug_print(f"Laden tickets data van: {tickets_path}", True)
        tickets_df = pd.read_csv(tickets_path, low_memory=False)
        debug_print(f"Tickets data geladen. Shape: {tickets_df.shape}", True)

        # Normaliseer event_name en zorg dat datums datetime zijn
        if 'event_name' in tickets_df.columns:
            tickets_df['event_name'] = tickets_df['event_name'].astype(str).apply(normalize_text)
        if 'first_event_date_start' in tickets_df.columns:
            tickets_df['first_event_date_start'] = pd.to_datetime(tickets_df['first_event_date_start'], errors='coerce', utc=True)
        if 'last_event_date_end' in tickets_df.columns:            
            tickets_df['last_event_date_end'] = pd.to_datetime(tickets_df['last_event_date_end'], errors='coerce', utc=True)
        if 'event_date' in tickets_df.columns:             
            tickets_df['event_date'] = pd.to_datetime(tickets_df['event_date'], errors='coerce', utc=True)
        if 'verkoopdatum' in tickets_df.columns: # Ensure 'verkoopdatum' is converted to datetime
            tickets_df['verkoopdatum'] = pd.to_datetime(tickets_df['verkoopdatum'], errors='coerce', utc=True)
        if 'city' in tickets_df.columns: # Normaliseer city voor consistentie
            tickets_df['city'] = tickets_df['city'].astype(str).apply(normalize_text)

    except FileNotFoundError:
        print(f"FOUT: Ticketdata bestand '{tickets_path}' niet gevonden. Kan niet doorgaan.")
        # Retourneer lege DataFrames om de tool niet te laten crashen, maar het zal niet werken.
        return pd.DataFrame(), None, pd.DataFrame() 
    except Exception as e:
        print(f"Kritieke FOUT bij laden/verwerken ticketdata van '{tickets_path}': {e}")
        debug_print(traceback.format_exc(), True)
        return pd.DataFrame(), None, pd.DataFrame()

    # 2. Artiesten Fallback Data
    artists_df_fb = pd.DataFrame()
    if os.path.exists(artists_fallback_path) and os.path.getsize(artists_fallback_path) > 0:
        debug_print(f"Laden fallback artiestendata van: {artists_fallback_path}", True)
        try:
            artists_df_fb = pd.read_csv(artists_fallback_path, low_memory=False)
            # Vind en normaliseer de artiestennaam kolom (vaak 'name' of 'artist')
            artist_col_fb = find_artist_column(artists_df_fb, ['name', 'artist'])
            if artist_col_fb:
                if artist_col_fb != 'artist': # Hernoem naar 'artist' voor interne consistentie
                    artists_df_fb.rename(columns={artist_col_fb: 'artist'}, inplace=True)
                artists_df_fb['artist'] = artists_df_fb['artist'].astype(str).apply(normalize_text)
            else:
                debug_print(f"WAARSCHUWING: Kon geen 'artist'/'name' kolom vinden in fallback '{artists_fallback_path}'.", True)
            
            # Zorg dat 'Date' kolom (indien aanwezig) een datetime object is met UTC timezone
            if 'Date' in artists_df_fb.columns:
                artists_df_fb['Date'] = pd.to_datetime(artists_df_fb['Date'], errors='coerce', utc=True)
            debug_print(f"Fallback artiestendata geladen. Shape: {artists_df_fb.shape}", True)
        except Exception as e:
            print(f"FOUT bij laden/verwerken fallback artiestendata van '{artists_fallback_path}': {e}")
            debug_print(traceback.format_exc(), True)
            # artists_df_fb blijft leeg bij een fout
    else:
        debug_print(f"INFO: Fallback artiestendata bestand '{artists_fallback_path}' niet gevonden of is leeg.", True)

    # 3. Historische Line-up Data (Optioneel, voor bijv. get_real_cities of andere analyses)
    lineup_df_hist = pd.DataFrame() # Initialiseer als lege DataFrame
    if lineup_path_historical and os.path.exists(lineup_path_historical) and os.path.getsize(lineup_path_historical) > 0 :
        debug_print(f"Laden historische line-up data van: {lineup_path_historical}", True)
        try:
            lineup_df_hist = pd.read_csv(lineup_path_historical, low_memory=False)
            # Normaliseer relevante kolommen indien aanwezig
            hist_lineup_artist_col = find_artist_column(lineup_df_hist)
            if hist_lineup_artist_col:
                lineup_df_hist[hist_lineup_artist_col] = lineup_df_hist[hist_lineup_artist_col].astype(str).apply(normalize_text)
            if 'event_name' in lineup_df_hist.columns:
                lineup_df_hist['event_name'] = lineup_df_hist['event_name'].astype(str).apply(normalize_text)
            debug_print(f"Historische line-up data geladen. Shape: {lineup_df_hist.shape}", True)
        except Exception as e:
            print(f"Fout bij laden/verwerken historische line-up data van '{lineup_path_historical}': {e}")
            debug_print(traceback.format_exc(), True)
            # lineup_df_hist blijft leeg
    else:
        debug_print(f"INFO: Historische line-up data bestand '{lineup_path_historical}' niet gevonden of leeg.", True)
        
    debug_print(f"Basisdata succesvol geladen.", True)
    return tickets_df, lineup_df_hist, artists_df_fb

# ===============================================================
# Feature Engineering Functie (Aangepast voor Live Tool & Spotify Focus)
# ===============================================================
def engineer_features(
    tickets: pd.DataFrame, 
    line_up_df: Optional[pd.DataFrame], # Van de input gemaakte line-up voor dit event
    artists_df: Optional[pd.DataFrame], # Live/fallback artiesten data voor deze line-up
    forecast_days: int, # Dit is de T-waarde van het geselecteerde model
    known_cities: Optional[List[str]],
    tickets_history_df: Optional[pd.DataFrame] = None, # HISTORICAL data for the event
    star_artist_percentile: float = 0.80, 
    is_prediction: bool = True, # Voor de live tool is dit altijd True
    verbose: bool = True
) -> pd.DataFrame:
    """
    Genereert features voor het voorspellingsmodel.
    Aangepast voor de live tool: werkt met input voor een enkel toekomstig event.
    Focus op Spotify-metrics voor artiesten.
    Kan nu historische ticketdata gebruiken voor featureberekening.
    """
    function_name = "engineer_features"
    debug_print(f"\n--- Start {function_name} (Live Tool Focus): Model T={forecast_days} ---", verbose)

    if tickets.empty:
        debug_print(f"FOUT in {function_name}: Input 'tickets' (df_events) DataFrame is leeg.", True)
        return pd.DataFrame()

    # Maak een kopie om de input DataFrame niet te wijzigen
    df_processed = tickets.copy()

    # --- Stap 1: Basis Voorbewerking (grotendeels al extern gebeurd voor live tool) ---
    # Event name normalisatie
    if 'event_name' in df_processed.columns:
        df_processed['event_name'] = df_processed['event_name'].astype(str).apply(normalize_text)
    else:
        debug_print(f"FOUT in {function_name}: 'event_name' ontbreekt in input 'tickets' DataFrame.", True)
        return pd.DataFrame() # Essentiële kolom

    current_event_name = df_processed['event_name'].iloc[0]      # voor debug-prints
    hist_data_for_calc  = pd.DataFrame()  

    # Datumkolommen (zouden al datetime en UTC moeten zijn van eerdere stappen)
    date_cols_to_check_ef = ['first_event_date_start', 'last_event_date_end', 'event_date'] # 'verkoopdatum' is niet relevant voor live toekomstig event
    for col_ef in date_cols_to_check_ef:
        if col_ef in df_processed.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_processed[col_ef]):
                df_processed[col_ef] = pd.to_datetime(df_processed[col_ef], errors='coerce', utc=True)
            elif df_processed[col_ef].dt.tz is None: # Als het datetime is maar naive
                 df_processed[col_ef] = df_processed[col_ef].dt.tz_localize('UTC')


    # City standaardisatie
    if 'city' in df_processed.columns and pd.notna(df_processed['city'].iloc[0]):
        normalized_city = normalize_text(df_processed['city'].iloc[0])
        # Markeer alleen als 'other' wanneer we een bekende steden-lijst hebben
        # én deze stad er echt niet in voorkomt.
        if known_cities and normalized_city not in known_cities:
            df_processed['buyer_city_normalized'] = 'other'
        else:
            df_processed['buyer_city_normalized'] = normalized_city
    else:
        df_processed['buyer_city_normalized'] = 'other'
    

    df_processed['days_until_event'] = forecast_days


    # --- Nieuwe Sectie: Voorbereiden en filteren van historische data ---
    historical_sales_for_event = pd.DataFrame()
    event_name_for_history_lookup = normalize_text(tickets['event_name'].iloc[0])

    if tickets_history_df is not None and not tickets_history_df.empty:
        debug_print(f"[{function_name}] Historische data meegegeven. Aantal rijen: {len(tickets_history_df)}", verbose)
        history_df_copy = tickets_history_df.copy()
        
        if 'event_name' in history_df_copy.columns:
            history_df_copy['event_name_normalized'] = history_df_copy['event_name'].astype(str).apply(normalize_text)
            event_specific_history = history_df_copy[history_df_copy['event_name_normalized'] == event_name_for_history_lookup]
            debug_print(f"[{function_name}] Rijen in history voor '{event_name_for_history_lookup}': {len(event_specific_history)}", verbose)

            if not event_specific_history.empty:
                # Zorg voor datetime types
                for date_col_hist in ['verkoopdatum', 'first_event_date_start', 'last_event_date_end']:
                    if date_col_hist in event_specific_history.columns:
                        event_specific_history[date_col_hist] = pd.to_datetime(event_specific_history[date_col_hist], errors='coerce', utc=True)
                    else:
                        debug_print(f"WAARSCHUWING: Kolom '{date_col_hist}' niet gevonden in historische data voor '{event_name_for_history_lookup}'.", True)
                
                event_specific_history.dropna(subset=['verkoopdatum', 'first_event_date_start'], inplace=True)

                if not event_specific_history.empty and 'first_event_date_start' in tickets.columns and pd.notna(tickets['first_event_date_start'].iloc[0]):
                    # Gebruik de event startdatum van de *te voorspellen instantie* (uit 'tickets' DataFrame)
                    current_event_start_date = pd.to_datetime(tickets['first_event_date_start'].iloc[0], errors='coerce', utc=True)
                    if pd.notna(current_event_start_date):
                        T_days_ago = current_event_start_date.normalize() - pd.Timedelta(days=forecast_days)
                        historical_sales_for_event = event_specific_history[event_specific_history['verkoopdatum'] < T_days_ago].copy()
                        debug_print(f"[{function_name}] Historische sales voor '{event_name_for_history_lookup}' voor T={forecast_days} (verkoopdatum < {T_days_ago.date()}): {len(historical_sales_for_event)} rijen.", verbose)
                    else:
                        debug_print(f"FOUT in {function_name}: Kon 'first_event_date_start' van het huidige event niet converteren naar datum voor T_days_ago berekening.", True)
                else:
                    debug_print(f"[{function_name}] Onvoldoende data in event_specific_history of 'tickets' voor T_days_ago berekening.", verbose)
        else:
            debug_print(f"WAARSCHUWING: Kolom 'event_name' niet gevonden in tickets_history_df.", True)
    else:
        debug_print(f"[{function_name}] Geen historische data (tickets_history_df) meegegeven of leeg.", verbose)


    # --- Stap 2: Target & Basis df_total (voor live tool is dit het enkele event) ---
    # df_total wordt hier effectief df_processed zelf, omdat we maar één event hebben.
    df_total_placeholder = df_processed[['event_name']].copy() # Alleen event_name nodig als basis
    # ── neem de gestandaardiseerde stad mee, anders verdwijnt hij ──
    df_total_placeholder['buyer_city_normalized'] = df_processed['buyer_city_normalized'].iloc[0]

    # ── Weer­kolommen alvast meenemen, zodat ze niet verloren gaan ──
    for _wcol in ['t_max', 't_min', 'rain_fall', 'max_wind']:
        df_total_placeholder[_wcol] = (
            df_processed[_wcol].iloc[0]     # waarde uit de 1-rij-input
            if _wcol in df_processed.columns else np.nan
    )


# --- Stap 2.5: Lag Features (sales_last_3_days) ---
    lag_days_sales_const = 3  # Definieer deze constante voor gebruik in Stap 2.5
    lag_sales_col_name = f'sales_last_{lag_days_sales_const}_days'
    log_lag_sales_col_name = f'log_{lag_sales_col_name}'
    
    sales_last_3_days_actual = 0.0 # Default

    # Gebruik event_specific_history (de volledige historie voor dit event)
    if 'event_specific_history' in locals() and not event_specific_history.empty and \
       'verkoopdatum' in event_specific_history.columns and \
       'first_event_date_start' in event_specific_history.columns and \
       pd.notna(current_event_start_date): # current_event_start_date is de startdatum van het te voorspellen event

        temp_hist_for_lag = event_specific_history.copy()
        # Bereken days_until_event_hist ten opzichte van de startdatum van het HUIDIGE te voorspellende event
        temp_hist_for_lag['days_until_event_hist_for_lag'] = \
            (pd.to_datetime(temp_hist_for_lag['first_event_date_start'], errors='coerce', utc=True).dt.normalize() - 
             pd.to_datetime(temp_hist_for_lag['verkoopdatum'], errors='coerce', utc=True).dt.normalize()).dt.days

        # Periode: T_day tot T_day + 2 (forecast_days, forecast_days+1, forecast_days+2 relatief aan event start)
        # Dit betekent days_until_event_hist >= forecast_days EN days_until_event_hist < forecast_days + lag_days_sales_const
        sales_in_lag_period_df = temp_hist_for_lag[
            (temp_hist_for_lag['days_until_event_hist_for_lag'] >= forecast_days) &
            (temp_hist_for_lag['days_until_event_hist_for_lag'] < forecast_days + lag_days_sales_const)
        ]
        sales_last_3_days_actual = float(len(sales_in_lag_period_df))
        debug_print(f"[{function_name}] Sales in lag periode T tot T+{lag_days_sales_const-1} (days_until_event_hist in [{forecast_days}...{forecast_days + lag_days_sales_const -1}]): {sales_last_3_days_actual} tickets.", verbose)
    else:
        debug_print(f"[{function_name}] Geen of onvoldoende historische data voor berekening sales_last_{lag_days_sales_const}_days.", verbose)

    df_total_placeholder[lag_sales_col_name] = sales_last_3_days_actual
    df_total_placeholder[log_lag_sales_col_name] = np.log1p(sales_last_3_days_actual)
    
    df_agg_base_for_merge = df_total_placeholder.copy()


    # --- Stap 3: Tijd-gebaseerde Features ---
    # Deze worden direct afgeleid van `first_event_date_start` uit de input `df_processed`.
    # We hebben maar één event, dus groupby is niet nodig voor deze specifieke features.
    event_time_features = pd.DataFrame(index=df_processed.index) # Maak lege df met zelfde index
    
    # Default event_duration_hours, kan overschreven worden door historie
    default_event_duration = 4 

    if 'first_event_date_start' in df_processed.columns and pd.notna(df_processed['first_event_date_start'].iloc[0]):
        event_datetime_obj = pd.to_datetime(df_processed['first_event_date_start'].iloc[0], errors='coerce', utc=True)
        if pd.notna(event_datetime_obj):
            event_time_features['day_of_week'] = event_datetime_obj.dayofweek
            event_time_features['event_month'] = event_datetime_obj.month
            event_time_features['event_year'] = event_datetime_obj.year
            event_time_features['is_weekend'] = event_time_features['day_of_week'].isin([5, 6]).astype(int)
            event_time_features['month_sin'] = np.sin(2 * np.pi * event_time_features['event_month'] / 12)
            event_time_features['month_cos'] = np.cos(2 * np.pi * event_time_features['event_month'] / 12)
            event_time_features['day_of_week_sin'] = np.sin(2 * np.pi * event_time_features['day_of_week'] / 7)
            event_time_features['day_of_week_cos'] = np.cos(2 * np.pi * event_time_features['day_of_week'] / 7)
            
            # Event duration
            if 'last_event_date_end' in df_processed.columns and pd.notna(df_processed['last_event_date_end'].iloc[0]):
                last_event_dt = pd.to_datetime(df_processed['last_event_date_end'].iloc[0], errors='coerce', utc=True)
                if pd.notna(last_event_dt) and last_event_dt > event_datetime_obj:
                    duration_h = (last_event_dt - event_datetime_obj).total_seconds() / 3600
                    event_time_features['event_duration_hours'] = min(max(1.0, duration_h), 30 * 24) # Cap
                else:
                    event_time_features['event_duration_hours'] = default_event_duration # Default
            else:
                event_time_features['event_duration_hours'] = default_event_duration # Default
            
            # Overschrijf event_duration_hours indien berekend uit historie
            if not historical_sales_for_event.empty and \
               'first_event_date_start' in historical_sales_for_event.columns and \
               'last_event_date_end' in historical_sales_for_event.columns:
                
                min_hist_start_date = historical_sales_for_event['first_event_date_start'].min()
                max_hist_end_date = historical_sales_for_event['last_event_date_end'].max()

                if pd.notna(min_hist_start_date) and pd.notna(max_hist_end_date) and max_hist_end_date > min_hist_start_date:
                    duration_h_hist = (max_hist_end_date - min_hist_start_date).total_seconds() / 3600
                    event_duration_hours_from_history = min(max(1.0, duration_h_hist), 30 * 24) # Cap
                    event_time_features['event_duration_hours'] = event_duration_hours_from_history
                    debug_print(f"[{function_name}] Event duration overridden by historical data: {event_duration_hours_from_history:.2f} uur.", verbose)
                else:
                    debug_print(f"[{function_name}] Kon event duration niet uit historie halen, gebruik live/default: {event_time_features['event_duration_hours'].iloc[0]:.2f} uur.", verbose)

        else: # Fallback als datumconversie mislukt
            for col_tf in ['day_of_week','event_month','event_year','is_weekend','month_sin','month_cos','day_of_week_sin','day_of_week_cos','event_duration_hours']:
                event_time_features[col_tf] = default_event_duration if 'duration' in col_tf else (0 if 'is_weekend' not in col_tf else -1)
    else: # Fallback als datum mist
        for col_tf in ['day_of_week','event_month','event_year','is_weekend','month_sin','month_cos','day_of_week_sin','day_of_week_cos','event_duration_hours']:
            event_time_features[col_tf] = default_event_duration if 'duration' in col_tf else (0 if 'is_weekend' not in col_tf else -1)

    # Sales duration (zal 0 zijn, geen 'verkoopdatum' in input voor live toekomstig event)
    event_time_features['sales_duration_hours_upto_t'] = 0
    event_time_features['log_sales_duration_hours_upto_t'] = np.log1p(0)

    # Merge tijd features met df_agg_base_for_merge
    if 'event_name' not in event_time_features.columns and 'event_name' in df_agg_base_for_merge.columns:
        event_time_features['event_name'] = df_agg_base_for_merge['event_name'].iloc[0] # Voeg event_name toe voor merge
    
    df_merged_with_features = pd.merge(df_agg_base_for_merge, event_time_features, on='event_name', how='left')

# --- Stap 4: Sales Features (cumulative, average daily) ---
    # Initialize with defaults
    df_merged_with_features['cumulative_sales_at_t'] = 0.0
    df_merged_with_features['log_cumulative_sales_at_t'] = 0.0
    df_merged_with_features['avg_daily_sales_before_t'] = 0.0
    df_merged_with_features['log_avg_daily_sales_before_t'] = 0.0
    avg_acceleration_val = 0.0   # fallback

    if not historical_sales_for_event.empty and 'verkoopdatum' in historical_sales_for_event.columns:
        tmp = historical_sales_for_event.copy()

        # dagen tot event t.o.v. dit event
        tmp['days_until_event_hist'] = (
            current_event_start_date.normalize() -
            tmp['verkoopdatum'].dt.normalize()
        ).dt.days

        daily_sales = (
            tmp.groupby(['days_until_event_hist'])
            .size()                               # tickets per dag
            .reset_index(name='daily_sales')
            .sort_values('days_until_event_hist')
        )

        if not daily_sales.empty:
            # diff() = dag-op-dag verandering  ➜ acceleration
            daily_sales['acceleration'] = (
                daily_sales['daily_sales']
                .diff()                     # dag t – dag t+1
                .fillna(0)
            )
            avg_acceleration_val = daily_sales['acceleration'].mean()

    df_merged_with_features['avg_acceleration'] = avg_acceleration_val


    if not historical_sales_for_event.empty:
        debug_print(f"[{function_name}] Berekenen cumulatieve/gemiddelde sales features uit {len(historical_sales_for_event)} historische transacties (tot T-dag).", verbose)
        
        cumulative_sales_at_t_val = float(len(historical_sales_for_event))
        df_merged_with_features['cumulative_sales_at_t'] = cumulative_sales_at_t_val
        df_merged_with_features['log_cumulative_sales_at_t'] = np.log1p(cumulative_sales_at_t_val)

        # Voor avg_daily_sales_before_t, gebruik de verkoopdatums binnen deze T-gefilterde historie
        if 'verkoopdatum' in historical_sales_for_event.columns and historical_sales_for_event['verkoopdatum'].nunique() > 0:
            # Bereken de span van verkoopdagen binnen de 'historical_sales_for_event' periode
            # Dit is de periode *tot* T dagen voor het event
            if historical_sales_for_event['verkoopdatum'].nunique() > 1:
                 sales_days_span = (historical_sales_for_event['verkoopdatum'].max() - historical_sales_for_event['verkoopdatum'].min()).days + 1
            else: # Als er maar op 1 dag is verkocht in deze periode
                 sales_days_span = 1
            
            if sales_days_span > 0:
                avg_daily_sales_val = cumulative_sales_at_t_val / sales_days_span
                df_merged_with_features['avg_daily_sales_before_t'] = avg_daily_sales_val
                df_merged_with_features['log_avg_daily_sales_before_t'] = np.log1p(avg_daily_sales_val)
            else:
                debug_print(f"[{function_name}] Sales day span (tot T-dag) is 0, avg_daily_sales_before_t blijft 0.", verbose)
        else:
            debug_print(f"[{function_name}] Niet genoeg unieke verkoopdatums in T-gefilterde historie voor avg_daily_sales.", verbose)
    else:
        debug_print(f"[{function_name}] Geen historische sales data (tot T-dag) voor cumulatieve/gemiddelde sales, blijven defaults (0.0).", verbose)

    # sales_last_3_days en log_sales_last_3_days zijn al in df_merged_with_features gezet door Stap 2.5
    # Controleer of ze bestaan en vul eventueel met 0 als ze er niet zijn (zou niet moeten als Stap 2.5 correct is)
    if 'sales_last_3_days' not in df_merged_with_features.columns:
        df_merged_with_features['sales_last_3_days'] = 0.0
    if 'log_sales_last_3_days' not in df_merged_with_features.columns:
        df_merged_with_features['log_sales_last_3_days'] = np.log1p(0.0)

    # --- Stap 5: Artist Feature Integratie (Focus op Spotify) ---
    debug_print(f"[{function_name}] Stap 5: Artiesten Features (Spotify Focus)...", verbose)
    
    lineup_artist_col = find_artist_column(line_up_df) 
    artist_col_in_artists_df = find_artist_column(artists_df, ['artist']) # artists_df komt van fetch_live_artist_data

    metrics_to_log_transform = {'SpotifyFollowers', 'SpotifyPopularity', 'num_artists'}
    aggregation_ops = ['mean', 'max', 'sum', 'std']
    
    active_metric_mappings = {} # { 'FeatureName': 'SourceColumnInArtistsDF' }
    if artists_df is not None: # Alleen mappen als artists_df bestaat
        if 'SpotifyPopularity' in artists_df.columns: active_metric_mappings['SpotifyPopularity'] = 'SpotifyPopularity'
        if 'SpotifyFollowers' in artists_df.columns: active_metric_mappings['SpotifyFollowers'] = 'SpotifyFollowers'
        for other_m in ['ChartmetricScore', 'TikTokViews', 'DeezerFans']: # Fallback metrics
            if other_m in artists_df.columns: active_metric_mappings[other_m] = other_m

    # Voorbereiden default artist features (een DataFrame met 1 rij voor het huidige event)
    default_artist_feature_cols = set(['num_artists', 'log_num_artists', 'has_artist_data', 'has_star_artist'])
    for metric_key_def in active_metric_mappings.keys():
        for agg_op_def in aggregation_ops:
            default_artist_feature_cols.add(f'{metric_key_def}_{agg_op_def}')
            if metric_key_def in metrics_to_log_transform:
                default_artist_feature_cols.add(f'log_{metric_key_def}_{agg_op_def}')
    
    # Maak een DataFrame met 1 rij, geïndexeerd zoals df_merged_with_features, gevuld met 0
    temp_default_artist_features_for_event = pd.DataFrame(0, index=df_merged_with_features.index, columns=list(default_artist_feature_cols))

    if artists_df is not None and not artists_df.empty and \
       line_up_df is not None and not line_up_df.empty and \
       lineup_artist_col and artist_col_in_artists_df:

        # Normalisatie van 'artist' kolom zou al gebeurd moeten zijn. Dubbelcheck:
        line_up_df[lineup_artist_col] = line_up_df[lineup_artist_col].astype(str).apply(normalize_text)
        artists_df[artist_col_in_artists_df] = artists_df[artist_col_in_artists_df].astype(str).apply(normalize_text)
        
        # Merge line-up (met event_name) met artiesten data.
        # line_up_df zou 'event_name' moeten hebben van predict_event_tickets_live
        if 'event_name' not in line_up_df.columns:
             line_up_df['event_name'] = df_merged_with_features['event_name'].iloc[0] # Voeg toe als het mist

        merged_artists_for_agg = pd.merge(line_up_df, artists_df, 
                                          left_on=lineup_artist_col, 
                                          right_on=artist_col_in_artists_df, 
                                          how='left', suffixes=('_lineup', '_artist'))
        
        if 'event_name_lineup' in merged_artists_for_agg.columns and 'event_name' not in merged_artists_for_agg.columns :
            merged_artists_for_agg.rename(columns={'event_name_lineup':'event_name'}, inplace=True)


        if verbose: debug_print(f"Merged artists for aggregation (shape {merged_artists_for_agg.shape}):\n{merged_artists_for_agg.head().to_string()}", verbose)

        if 'event_name' in merged_artists_for_agg.columns and not merged_artists_for_agg.empty:
            agg_ops_to_apply = {'num_artists': (lineup_artist_col, 'nunique')}
            metrics_actually_aggregated = {}

            for feat_map_key, source_col_map_val in active_metric_mappings.items():
                if source_col_map_val in merged_artists_for_agg.columns:
                    merged_artists_for_agg[source_col_map_val] = pd.to_numeric(merged_artists_for_agg[source_col_map_val], errors='coerce')
                    metrics_actually_aggregated[feat_map_key] = source_col_map_val
                    for agg_op_str in aggregation_ops:
                        agg_ops_to_apply[f'{feat_map_key}_{agg_op_str}'] = (source_col_map_val, agg_op_str)
            
            if metrics_actually_aggregated:
                try:
                    aggregated_artist_features_event = merged_artists_for_agg.groupby('event_name').agg(**agg_ops_to_apply).reset_index()

                    if not aggregated_artist_features_event.empty:
                        # Log transformaties, has_artist_data, has_star_artist
                        if 'num_artists' in aggregated_artist_features_event.columns and 'num_artists' in metrics_to_log_transform:
                            aggregated_artist_features_event['log_num_artists'] = np.log1p(aggregated_artist_features_event['num_artists'].fillna(0))

                        for metric_log_key in metrics_actually_aggregated.keys():
                            if metric_log_key in metrics_to_log_transform:
                                for agg_log_op in aggregation_ops:
                                    col_orig_log = f'{metric_log_key}_{agg_log_op}'
                                    if col_orig_log in aggregated_artist_features_event.columns:
                                        aggregated_artist_features_event[f'log_{col_orig_log}'] = np.log1p(aggregated_artist_features_event[col_orig_log].fillna(0))
                        
                        star_metric_col_name = 'SpotifyFollowers_max'
                        log_star_metric_col_name = f'log_{star_metric_col_name}'
                        if star_metric_col_name in aggregated_artist_features_event.columns:
                            aggregated_artist_features_event['has_artist_data'] = (aggregated_artist_features_event[star_metric_col_name].fillna(0) > 0).astype(int)
                            if log_star_metric_col_name in aggregated_artist_features_event.columns:
                                valid_star_data_series = aggregated_artist_features_event.loc[aggregated_artist_features_event['has_artist_data'] == 1, log_star_metric_col_name].dropna()
                                star_q_threshold = valid_star_data_series.quantile(star_artist_percentile) if not valid_star_data_series.empty else 0
                                aggregated_artist_features_event['has_star_artist'] = (aggregated_artist_features_event[log_star_metric_col_name].fillna(0) > star_q_threshold).astype(int)
                            else: aggregated_artist_features_event['has_star_artist'] = 0
                        else:
                            aggregated_artist_features_event['has_artist_data'] = 0; aggregated_artist_features_event['has_star_artist'] = 0
                        
                        # Zorg dat event_name in aggregated_artist_features_event ook genormaliseerd is.
                        aggregated_artist_features_event['event_name'] = aggregated_artist_features_event['event_name'].astype(str).apply(normalize_text)
                        df_merged_with_features = pd.merge(df_merged_with_features, aggregated_artist_features_event, on='event_name', how='left')
                    else: # aggregated_artist_features_event is leeg na groupby
                         debug_print("Aggregatie artiest features resulteerde in lege DataFrame. Gebruik defaults.", verbose)
                         df_merged_with_features = pd.concat([df_merged_with_features, temp_default_artist_features_for_event.drop(columns=['event_name'], errors='ignore')], axis=1)


                except Exception as e_agg_final:
                    debug_print(f"FOUT bij artiestenaggregatie voor event: {e_agg_final}\n{traceback.format_exc()}", True)
                    df_merged_with_features = pd.concat([df_merged_with_features, temp_default_artist_features_for_event.drop(columns=['event_name'], errors='ignore')], axis=1)
            else:
                debug_print("Geen actieve metrics voor aggregatie. Gebruik defaults voor artiesten.", verbose)
                df_merged_with_features = pd.concat([df_merged_with_features, temp_default_artist_features_for_event.drop(columns=['event_name'], errors='ignore')], axis=1)
        else:
            debug_print("Kan artiesten niet aggregeren (geen event_name of data). Gebruik defaults.", verbose)
            df_merged_with_features = pd.concat([df_merged_with_features, temp_default_artist_features_for_event.drop(columns=['event_name'], errors='ignore')], axis=1)
    else:
        debug_print("Geen artiesten- of line-up data beschikbaar. Gebruik defaults voor artiesten.", verbose)
        df_merged_with_features = pd.concat([df_merged_with_features, temp_default_artist_features_for_event.drop(columns=['event_name'], errors='ignore')], axis=1)

    # Zorg dat alle default artist feature kolommen bestaan in de uiteindelijke df_merged_with_features
    for col_art_final in default_artist_feature_cols:
        if col_art_final not in df_merged_with_features.columns:
            df_merged_with_features[col_art_final] = 0 # Voeg toe met 0 als het mist
        elif pd.api.types.is_numeric_dtype(df_merged_with_features[col_art_final]): # Als het bestaat, vul NaN met 0
            df_merged_with_features[col_art_final] = df_merged_with_features[col_art_final].fillna(0)


    # --- Stap 6, 7, 8 (Demografie, Ticket Info, Max Capacity) ---
    debug_print(f"[{function_name}] Starten met Demografie, Ticket Info, Max Capacity...", verbose)


    # Deze worden mogelijk overschreven als er historische data is.
    demo_ticket_cols_to_init = {
        'avg_age': 30.0, 'std_age': 0.0,
        'male_percentage': 0.0, 'female_percentage': 0.0, 'nb_percentage': 0.0,
        'unique_cities': 0.0, 'log_unique_cities': 0.0,
        'main_city_buyer_ratio': 0.0,
        'avg_product_value': 0.0, 'std_product_value': 0.0,
        'avg_total_price': 0.0, 'scanned_percentage': 0.0
    }
    for col, default_val in demo_ticket_cols_to_init.items():
        # Als de kolom al bestaat (bijv. uit df_processed), behoud die waarde, anders initialiseer.
        # df_merged_with_features is een kopie van df_processed (de 1-rij context).
        if col not in df_merged_with_features.columns:
            df_merged_with_features[col] = default_val
        else: # Vul NaN met default als het al bestond maar NaN was
             df_merged_with_features[col] = df_merged_with_features[col].fillna(default_val)


    # historical_sales_for_event is de historie voor dit event, al gefilterd op verkoopdatum < T_days_ago
    # (dus, verkopen die plaatsvonden *tot* T dagen voor het event)
    if historical_sales_for_event is not None and not historical_sales_for_event.empty:
        debug_print(f"[{function_name}] Berekenen demo/ticket features uit {len(historical_sales_for_event)} historische transacties (tot T-dag).", verbose)
        
        # Initialize hist_data_for_calc before using it
        hist_data_for_calc = historical_sales_for_event.copy()
        
        # Now we can use hist_data_for_calc
        if 'city' in hist_data_for_calc.columns:
            hist_data_for_calc['buyer_city_normalized'] = hist_data_for_calc['city'].astype(str).apply(normalize_text)
            hist_data_for_calc['buyer_city_standardized'] = hist_data_for_calc['buyer_city_normalized'].apply(
                lambda x: x if not known_cities or x in known_cities else 'other'
            )
        else:
            hist_data_for_calc['buyer_city_normalized'] = 'other'
            hist_data_for_calc['buyer_city_standardized'] = 'other'

        # Gemiddelde Leeftijd & Standaarddeviatie
        if 'age' in hist_data_for_calc.columns and hist_data_for_calc['age'].notna().any():
            df_merged_with_features['avg_age'] = hist_data_for_calc['age'].mean()
            df_merged_with_features['std_age'] = hist_data_for_calc['age'].std()
        else:
            debug_print(f"[{function_name}] Kolom 'age' mist of heeft geen data in historie voor {current_event_name}.", verbose)
            # Defaults zijn al gezet, geen actie nodig

        # Gender Percentages
        if 'gender' in hist_data_for_calc.columns and hist_data_for_calc['gender'].notna().any():
            gender_counts = hist_data_for_calc['gender'].str.lower().value_counts(normalize=True)
            df_merged_with_features['male_percentage'] = gender_counts.get('man', gender_counts.get('male', 0.0)) * 100
            df_merged_with_features['female_percentage'] = gender_counts.get('vrouw', gender_counts.get('woman', gender_counts.get('female', 0.0))) * 100
            df_merged_with_features['nb_percentage'] = gender_counts.get('nonbinary', 0.0) * 100
        else:
            debug_print(f"[{function_name}] Kolom 'gender' mist of heeft geen data in historie voor {current_event_name}.", verbose)

        # Unieke Steden & Main City Buyer Ratio
        # df_merged_with_features['buyer_city_normalized'] zou de gestandaardiseerde stad van het event zelf moeten bevatten.
        if 'city' in hist_data_for_calc.columns and \
           'buyer_city_normalized' in df_merged_with_features.columns and \
           pd.notna(df_merged_with_features['buyer_city_normalized'].iloc[0]):
            
 

            city_counts = hist_data_for_calc.groupby('buyer_city_standardized').size().reset_index(name='count') # Group by buyer cities
            if not city_counts.empty:
                city_counts = city_counts.sort_values(['count'], ascending=[False])
                main_city_from_buyers = city_counts.iloc[0]['buyer_city_standardized']
                main_city_from_buyers_count = city_counts.iloc[0]['count']
                
                total_buyers_in_hist = len(hist_data_for_calc) # Total historical transactions considered

                if total_buyers_in_hist > 0:
                    df_merged_with_features['main_city_buyer_ratio'] = main_city_from_buyers_count / total_buyers_in_hist
                    debug_print(f"[{function_name}] Main buyer city ({main_city_from_buyers}) ratio: {main_city_from_buyers_count}/{total_buyers_in_hist} = {df_merged_with_features['main_city_buyer_ratio'].iloc[0]:.4f}", verbose)
   
            
            valid_buyer_cities = hist_data_for_calc['buyer_city_standardized'].dropna()
            if not valid_buyer_cities.empty:
                df_merged_with_features['unique_cities'] = float(valid_buyer_cities.nunique())
                # Gebruik .iloc[0] omdat df_merged_with_features 1 rij heeft
                df_merged_with_features['log_unique_cities'] = np.log1p(df_merged_with_features['unique_cities'].iloc[0])

            else:
                debug_print(f"[{function_name}] Geen valide kopersteden in historie voor unique_cities etc.", verbose)
        else: # De hoofconditie (if 'city' in hist_data_for_calc.columns and 'buyer_city_normalized' in df_merged_with_features ...) is niet voldaan
            debug_print(f"[{function_name}] Overgeslagen: Stadsfeatures (incl. main_city_buyer_ratio) niet berekend. Controleer of 'city' kolom aanwezig is in historische data (hist_data_for_calc) EN 'buyer_city_normalized' in de event data (df_merged_with_features).", verbose)
            # main_city_buyer_ratio blijft de default waarde (0.0) die eerder is geïnitialiseerd

        # ───────────────────────────────────────────────────────────────
        # Product- en order-waarde features
        # ───────────────────────────────────────────────────────────────
        prod_val_col_options = ['product_value', 'order_waarde_item_actueel']
        qty_col_options      = ['ticket_quantity', 'aantal', 'qty', 'quantity']  

        used_val_col = next(
            (col for col in prod_val_col_options
            if col in hist_data_for_calc.columns and hist_data_for_calc[col].notna().any()),
            None
        )

        if used_val_col:
            # maak kopie & forceer numeriek
            val_series = pd.to_numeric(hist_data_for_calc[used_val_col], errors='coerce').dropna()

            # 1) Detecteer of de waarden waarschijnlijk in centen staan
            if val_series.median() > 250:                # arbitrair: > € 2,50 betekent vaak centen
                val_series = val_series / 100.0


            # 2) Indien een quantity-kolom bestaat → bereken prijs per ticket
            used_qty_col = next((c for c in qty_col_options if c in hist_data_for_calc.columns), None)
            if used_qty_col:
                qty_series = pd.to_numeric(hist_data_for_calc[used_qty_col], errors='coerce').fillna(1)
                # zorg dat lengtes gelijk blijven
                val_series = val_series.reset_index(drop=True) / qty_series.reset_index(drop=True).replace(0, 1)

            val_series = val_series[val_series > 0] 

            # 3) Filter extreme outliers (> € 200) – optioneel, maar voorkomt scheve gemiddelden
            val_series = val_series[val_series < 200]

            if not val_series.empty:
                df_merged_with_features['avg_product_value'] = val_series.mean()
                df_merged_with_features['std_product_value'] = val_series.std()
                debug_print(
                    f"[{function_name}] avg_product_value berekend uit {len(val_series)} records: "
                    f"{val_series.mean():.2f} €", verbose
                )
            else:
                debug_print(
                    f"[{function_name}] Geen valide productwaarden na filtering; defaults blijven staan.",
                    verbose
                )
        else:
            debug_print(f"[{function_name}] Geen product-value kolom gevonden; defaults blijven staan.", verbose)
        # ───────────────────────────────────────────────────────────────

        if 'total_price' in hist_data_for_calc.columns and hist_data_for_calc['total_price'].notna().any():
            hist_data_for_calc['total_price'] = pd.to_numeric(hist_data_for_calc['total_price'], errors='coerce')
            if hist_data_for_calc['total_price'].notna().any():
                 df_merged_with_features['avg_total_price'] = hist_data_for_calc['total_price'].mean()
        else:
            debug_print(f"[{function_name}] Geen 'total_price' in historie.", verbose)

        if 'product_is_scanned' in hist_data_for_calc.columns and hist_data_for_calc['product_is_scanned'].notna().any():
            hist_data_for_calc['product_is_scanned'] = pd.to_numeric(hist_data_for_calc['product_is_scanned'], errors='coerce')
            if hist_data_for_calc['product_is_scanned'].notna().any():
                 df_merged_with_features['scanned_percentage'] = hist_data_for_calc['product_is_scanned'].mean()
        else:
            debug_print(f"[{function_name}] Geen 'product_is_scanned' in historie.", verbose)
    else: # Als historical_sales_for_event leeg of None was
        debug_print(f"[{function_name}] Geen historische data (tot T-dag) voor demo/ticket features, blijven defaults.", verbose)

    # Vul eventuele resterende NaNs (die door mean/std op lege series kunnen ontstaan) met 0.0
    # De initialisatie aan het begin van deze sectie heeft al defaults gezet.
    # Deze loop zorgt ervoor dat als een berekening NaN opleverde, het 0.0 wordt.
    for col_to_fill_final in demo_ticket_cols_to_init.keys():
        if col_to_fill_final in df_merged_with_features.columns:
            # df_merged_with_features is een 1-rij DataFrame. Gebruik .iloc[0] voor assignment.
            current_val = df_merged_with_features[col_to_fill_final].iloc[0]
            if pd.isna(current_val):
                df_merged_with_features.loc[df_merged_with_features.index[0], col_to_fill_final] = 0.0
        else: 
            df_merged_with_features[col_to_fill_final] = 0.0


    # Max capacity 
    if 'max_capacity' not in df_merged_with_features.columns or pd.isna(df_merged_with_features['max_capacity'].iloc[0]):
        df_merged_with_features['max_capacity'] = 1700.0 
    # Vul NaN specifiek voor log transformatie als max_capacity NaN was en nu 1700 is.
    df_merged_with_features['log_max_capacity'] = np.log1p(df_merged_with_features['max_capacity'].fillna(1700.0))


    # --- Stap 9: Weer Features ---
    # De weerdata (t_max, t_min, etc.) zit al in df_merged_with_features (vanuit de input 'tickets' DataFrame).
    # Zorg dat ze bestaan en numeriek zijn, anders vul met defaults.

    
    weather_cols_for_model = ['t_max', 't_min', 'rain_fall', 'max_wind']
    for wc_model_col in weather_cols_for_model:
        if wc_model_col not in df_merged_with_features.columns:
            df_merged_with_features[wc_model_col] = 0.0 # Bijv. 15 voor t_max, 5 voor t_min, 0 voor rain/wind
        elif pd.isna(df_merged_with_features[wc_model_col].iloc[0]):
             df_merged_with_features[wc_model_col] = df_merged_with_features[wc_model_col].fillna(0.0) # Vul NaN met default
        # Zorg dat het numeriek is
        df_merged_with_features[wc_model_col] = pd.to_numeric(df_merged_with_features[wc_model_col], errors='coerce').fillna(0.0)


    # --- Stap 10: Finale Check & Toevoegen forecast_days ---
    df_merged_with_features['forecast_days'] = forecast_days # Dit is de T van het model
    

    debug_print(f"Feature engineering voltooid. Shape output: {df_merged_with_features.shape}", verbose)
    if verbose and not df_merged_with_features.empty:
        # Check op constante kolommen (behalve 'forecast_days' en 'event_name' als die constant is voor 1 event)
        cols_to_check_constancy = [c for c in df_merged_with_features.columns if c not in ['event_name', 'forecast_days']]
        constant_cols_found = [col for col in cols_to_check_constancy if df_merged_with_features[col].nunique(dropna=False) <= 1]
        if constant_cols_found: 
            debug_print(f"WAARSCHUWING: Potentieel constante kolommen na feature engineering: {constant_cols_found}", verbose)
        # debug_print(f"Voorbeeld van features voor voorspelling:\n{df_merged_with_features.head().to_string()}", verbose)

    return df_merged_with_features

# ===============================================================
# Functie voor ophalen Toekomstige Evenementen
# ===============================================================
def get_future_events(tickets_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Haalt alleen de toekomstige evenementen op uit de tickets dataframe.
    Zorgt ervoor dat 'first_event_date_start' een timezone-aware (UTC) datetime object is.
    Berekent 'days_until_event'.
    """
    if tickets_df.empty:
        debug_print("Input tickets_df voor get_future_events is leeg.", verbose)
        return pd.DataFrame()
    
    if 'first_event_date_start' not in tickets_df.columns:
        debug_print("FOUT: Kolom 'first_event_date_start' niet gevonden in tickets_df voor get_future_events.", True)
        return pd.DataFrame()

    # Werk met een kopie om de originele DataFrame niet te wijzigen
    df = tickets_df.copy()

    # Zorg ervoor dat de datum kolom een timezone-aware datetime type is (UTC)
    if not pd.api.types.is_datetime64_any_dtype(df['first_event_date_start']) or df['first_event_date_start'].dt.tz is None:
        df['first_event_date_start'] = pd.to_datetime(df['first_event_date_start'], errors='coerce', utc=True)
    
    # Verwijder rijen waar datumconversie mislukt is
    df.dropna(subset=['first_event_date_start'], inplace=True)
    if df.empty:
        debug_print("Geen valide datums over na conversie in get_future_events.", verbose)
        return pd.DataFrame()

    # Huidige datum, ook timezone-aware (UTC) en genormaliseerd naar middernacht
    today_utc = pd.Timestamp.now(tz='UTC').normalize()
    
    # Filter alleen events in de toekomst (vergelijk genormaliseerde datums)
    # Normaliseer event datums ook naar middernacht voor een eerlijke vergelijking
    future_mask = df['first_event_date_start'].dt.normalize() > today_utc
    future_events_df = df[future_mask].copy() # .copy() om SettingWithCopyWarning te vermijden
    
    if future_events_df.empty:
        debug_print("Geen toekomstige evenementen gevonden na filteren.", verbose)
        return pd.DataFrame()
        
    # Bereken aantal dagen tot event (verschil tussen twee genormaliseerde UTC datums)
    future_events_df['days_until_event'] = (future_events_df['first_event_date_start'].dt.normalize() - today_utc).dt.days
    
    # Haal unieke evenementen op (op basis van genormaliseerde event_name) en sorteer deze op datum
    # Zorg dat event_name genormaliseerd is voor drop_duplicates
    if 'event_name' in future_events_df.columns:
        future_events_df['event_name_normalized_temp'] = future_events_df['event_name'].astype(str).apply(normalize_text)
        unique_future_events = future_events_df.sort_values('first_event_date_start').drop_duplicates('event_name_normalized_temp', keep='first')
        unique_future_events = unique_future_events.drop(columns=['event_name_normalized_temp'])
    else: # Fallback als event_name kolom er niet is (zou niet moeten)
        debug_print("WAARSCHUWING: 'event_name' kolom niet in future_events_df, kan niet optimaal dedupliceren.", verbose)
        unique_future_events = future_events_df.sort_values('first_event_date_start') # Geen deduplicatie

    debug_print(f"{len(unique_future_events)} unieke toekomstige evenementen gevonden.", verbose)
    return unique_future_events

# ===============================================================
# Hoofd Voorspellingsfunctie (Interne Wrapper die engineer_features gebruikt)
# ===============================================================
def predict_tickets( 
    df_events: pd.DataFrame,    # Minimale DataFrame met event info (+ weer) voor het te voorspellen event
    T_model: int,               # Geselecteerde T-waarde van het te gebruiken model
    known_cities: Optional[List[str]],
    line_up_df: Optional[pd.DataFrame], # On-the-fly gemaakte line-up DataFrame
    artists_df: Optional[pd.DataFrame], # Live/fallback artiesten data DataFrame
    tickets_history_df: Optional[pd.DataFrame] = None, # HISTORICAL data for the event
    verbose: bool = False
) -> pd.DataFrame:
    """
    Interne functie die features genereert en een voorspelling doet met het geselecteerde model.
    """
    global best_models
    
    if T_model not in best_models:
        # Dit zou niet moeten gebeuren als get_closest_trained_model_T correct werkt
        debug_print(f"FOUT in predict_tickets: Model voor T={T_model} is niet geladen in 'best_models'.", True)
        return pd.DataFrame()
            
    model_pipeline_to_use, target_is_transformed = best_models[T_model]
    
    # Genereer features met de (mogelijk live) input DataFrames
    # De 'forecast_days' parameter voor engineer_features is de T van het model
    features_for_model_df = engineer_features(
        tickets=df_events,       # DataFrame met event info (+ weer) voor het te voorspellen event
        line_up_df=line_up_df,
        artists_df=artists_df,
        forecast_days=T_model, # Gebruik de T van het geselecteerde model
        known_cities=known_cities,
        tickets_history_df=tickets_history_df, # Pass historical data
        is_prediction=True,      # Altijd True voor deze flow
        verbose=verbose
    )
    
    if features_for_model_df.empty:
        debug_print("FOUT: Feature engineering resulteerde in een lege DataFrame. Kan niet voorspellen.", True)
        return pd.DataFrame()

    # Voorspelling doen
    try:
        X_to_predict = features_for_model_df.copy()
        event_names_for_output = pd.Series(["unknown_event"] * len(X_to_predict)) # Default
        if 'event_name' in X_to_predict.columns:
            event_names_for_output = X_to_predict['event_name'] 
            X_to_predict = X_to_predict.drop(columns=['event_name'], errors='ignore')
        
        # preprocessor_step is de eerste stap in de pipeline die feature_names_in_ heeft
        preprocessor_step = None
        if hasattr(model_pipeline_to_use, 'named_steps') and 'preprocessor' in model_pipeline_to_use.named_steps:
            preprocessor_step = model_pipeline_to_use.named_steps['preprocessor']
        elif len(model_pipeline_to_use.steps) > 0 and hasattr(model_pipeline_to_use.steps[0][1], 'feature_names_in_'): # Eerste stap is vaak preprocessor
            preprocessor_step = model_pipeline_to_use.steps[0][1]

        if preprocessor_step and hasattr(preprocessor_step, 'feature_names_in_'):
            expected_model_input_features = list(preprocessor_step.feature_names_in_)
            
            # Maak een lege DataFrame met de verwachte kolommen en vul met 0
            # Dit zorgt voor de juiste volgorde en vult missende kolommen met 0.
            aligned_X_to_predict = pd.DataFrame(columns=expected_model_input_features, index=X_to_predict.index)
            aligned_X_to_predict = aligned_X_to_predict.fillna(0.0) # Vul met float 0.0

            # Vul met waarden uit X_to_predict waar kolommen overeenkomen
            common_cols = [col for col in X_to_predict.columns if col in expected_model_input_features]
            aligned_X_to_predict[common_cols] = X_to_predict[common_cols]
            
            # Identificeer en rapporteer kolommen in X_to_predict die niet verwacht worden
            extra_cols_found = set(X_to_predict.columns) - set(expected_model_input_features)
            if extra_cols_found:
                debug_print(f"WAARSCHUWING: Extra kolommen in data voor voorspelling die niet in model's input features zaten: {extra_cols_found}. Deze worden genegeerd.", verbose)
            
            X_to_predict_final = aligned_X_to_predict[expected_model_input_features] # Zorg voor juiste volgorde
        else:
            debug_print("WAARSCHUWING: Kon verwachte input features van preprocessor niet bepalen. Gebruik alle features uit engineer_features.", verbose)
            X_to_predict_final = X_to_predict # Minder robuust

        if X_to_predict_final.empty and not features_for_model_df.empty : 
             debug_print("FOUT: X_to_predict_final is leeg geworden na kolomselectie/reindexen.", True)
             return pd.DataFrame()
        elif X_to_predict_final.empty: 
             debug_print("FOUT: X_to_predict_final is leeg voor model input.", True)
             return pd.DataFrame()

        if verbose: 
            debug_print(f"Input shape voor model_pipeline.predict: {X_to_predict_final.shape}", verbose)

            print(f"[DEBUG] SCRIPT A: Input shape voor model_pipeline.predict: {X_to_predict_final.shape}")
            print(f"[DEBUG] SCRIPT A: Kolommen voor model_pipeline.predict: {X_to_predict_final.columns.tolist()}")
            print(f"[DEBUG] SCRIPT A: Feature WAARDEN voor model_pipeline.predict:\n{X_to_predict_final.to_string()}")

        raw_predictions = model_pipeline_to_use.predict(X_to_predict_final)
        
        
        # --- TERUGTRANSFORMATIE ---
        predictions_after_transform = np.expm1(raw_predictions) if target_is_transformed else raw_predictions
        
        # Zorg dat voorspellingen niet negatief zijn na transformatie
        predictions_non_negative = np.maximum(0, predictions_after_transform)

        # Haal cumulative_sales_at_t op uit de originele feature set voor de aanpassing
        # features_for_model_df bevat de features VOORDAT ze werden afgestemd op model input.
        # Het is belangrijk dat de index overeenkomt met de voorspellingen.
        final_adjusted_predictions = predictions_non_negative # Start met niet-negatieve voorspellingen

        if 'cumulative_sales_at_t' in features_for_model_df.columns:
            # Zorg ervoor dat we werken met een numpy array voor cumulative_sales voor gemakkelijke broadcasting/vergelijking
            cumulative_sales_values = features_for_model_df['cumulative_sales_at_t'].values
            
            # Zorg ervoor dat de lengtes overeenkomen 
            # (zou moeten als features_for_model_df de basis was voor X_to_predict_final en de indexen zijn behouden)
            if len(cumulative_sales_values) == len(predictions_non_negative):
                final_adjusted_predictions = np.maximum(predictions_non_negative, cumulative_sales_values)
                if verbose: # Check of verbose beschikbaar is in deze scope, anders moet het als parameter mee.
                    adjusted_count = (final_adjusted_predictions > predictions_non_negative).sum()
                    if adjusted_count > 0:
                        debug_print(f"   INFO (predict_tickets): {adjusted_count} voorspelling(en) aangepast omdat de oorspronkelijke voorspelling lager was dan cumulative_sales_at_t.", verbose)
            else:
                debug_print(f"WAARSCHUWING (predict_tickets): Mismatch in lengte tussen voorspellingen ({len(predictions_non_negative)}) en cumulative_sales_at_t ({len(cumulative_sales_values)}). Aanpassing o.b.v. cumulatieve sales niet volledig uitgevoerd.", verbose)
                # Fallback: gebruik de predictions_non_negative als er een mismatch is.
        else:
            debug_print("WAARSCHUWING (predict_tickets): 'cumulative_sales_at_t' kolom niet gevonden in features_for_model_df. Kan voorspellingen niet aanpassen aan cumulatieve sales.", verbose)

        # Afronden op gehele getallen
        final_predictions_rounded = np.round(final_adjusted_predictions).astype(int)
        
        prediction_results_df = pd.DataFrame({
            'event_name': event_names_for_output, 
            'predicted_full_event_tickets': final_predictions_rounded
        })
        return prediction_results_df
    
    except KeyError as e_key:
        debug_print(f"FOUT (KeyError) predict_tickets: Kolom {e_key} niet gevonden. Mismatch features vs model?", True)
        debug_print(traceback.format_exc(), True)
    except ValueError as e_val:
        debug_print(f"FOUT (ValueError) predict_tickets: {e_val}. Check data types/shapes/NaNs.", True)
        debug_print(traceback.format_exc(), True)
    except Exception as e_generic_pred:
        debug_print(f"ALGEMENE FOUT predict_tickets: {e_generic_pred}", True)
        debug_print(traceback.format_exc(), True)
    return pd.DataFrame()

# ===============================================================
# Hoofdfunctie voor Live Voorspellingen (gebruikt door interactive_tool)
# ===============================================================
def predict_event_tickets_live(
    selected_event_name: str,
    selected_event_data: pd.Series, 
    manual_lineup_str: Optional[str],
    base_tickets_df_context: pd.DataFrame, # This is the historical data
    base_artists_df_for_fallback: pd.DataFrame, 
    known_cities_context: Optional[List[str]], # Added known_cities_context parameter
    verbose: bool = True
) -> Tuple[int, str]:
    """
    Maakt een voorspelling voor een specifiek evenement, met live data.
    """
    global best_models 
    if not best_models:
        return 0, "FOUT: Modellen zijn niet geladen."

    normalized_event_name_internal = normalize_text(selected_event_name)
    event_date_val = selected_event_data.get('first_event_date_start')
    if pd.isna(event_date_val):
        return 0, f"FOUT: Geen startdatum voor '{selected_event_name}'."
    try:
        event_date_dt_utc_internal = pd.to_datetime(event_date_val, errors='raise', utc=True).normalize()
    except Exception as e_date_conv_live:
        debug_print(f"FOUT: Kon startdatum '{event_date_val}' niet converteren: {e_date_conv_live}", True)
        return 0, f"FOUT: Ongeldige startdatum voor '{selected_event_name}'."



    current_artists_df_for_event = pd.DataFrame()
    current_lineup_df_for_event = pd.DataFrame()

    if manual_lineup_str and manual_lineup_str.strip():
        artists_names_from_input = [artist.strip() for artist in manual_lineup_str.split(',') if artist.strip()]
        if artists_names_from_input:
            debug_print(f"Ophalen artiestendata voor: {', '.join(artists_names_from_input)}", verbose)
            current_artists_df_for_event = fetch_live_artist_data_for_lineup(
                artists_names_from_input, base_artists_df_for_fallback, verbose)
            
            normalized_artist_names_for_df = [normalize_text(a) for a in artists_names_from_input]
            current_lineup_df_for_event = pd.DataFrame({
                'event_name': [normalized_event_name_internal] * len(normalized_artist_names_for_df),
                'event_date': [event_date_dt_utc_internal] * len(normalized_artist_names_for_df), 
                'artist': normalized_artist_names_for_df
            })
            if 'event_date' in current_lineup_df_for_event:
                 current_lineup_df_for_event['event_date'] = pd.to_datetime(current_lineup_df_for_event['event_date'], errors='coerce', utc=True)
    else:
        debug_print("Geen line-up opgegeven.", verbose)
        artist_cols_default = ['artist', 'Date', 'SpotifyPopularity', 'SpotifyFollowers', 'ChartmetricScore', 'TikTokViews', 'DeezerFans', 'genres_list', 'spotify_id', 'spotify_url']
        current_artists_df_for_event = pd.DataFrame(columns=artist_cols_default)
        lineup_cols_default = ['event_name', 'event_date', 'artist']
        current_lineup_df_for_event = pd.DataFrame(columns=lineup_cols_default)

    weather_data_dict_for_event = {}
    weather_cols_for_model = ['t_max', 't_min', 'rain_fall', 'max_wind'] 
 
    # 1) bepaal event-stad
    event_city_for_weather = (
            selected_event_data.get("event_city")    # voor later, eventueel met andere steden  
            or "Amsterdam"                                
    )

    # 2) haal weer op
    live_weather_api_data = fetch_live_weather_for_event_day(
            event_date_dt_utc_internal,
            event_city_for_weather,
            verbose
    )
    weather_source_info = "Basis (CSV)"
    if live_weather_api_data:
        debug_print(f"Live weerdata (API) gebruikt voor {selected_event_name}.", verbose)
        weather_source_info = "Live (API)"
        for col_name_w in weather_cols_for_model:
            weather_data_dict_for_event[col_name_w] = [live_weather_api_data.get(col_name_w, np.nan)]
    else:
        debug_print(f"Fallback weerdata (uit CSV of defaults) wordt gebruikt voor {selected_event_name}.", verbose)
        weather_defaults_live = {'t_max': 15.0, 't_min': 5.0, 'rain_fall': 0.0, 'max_wind': 0.0}
        for col_name_w in weather_cols_for_model:
            csv_val = selected_event_data.get(col_name_w)
            if col_name_w in ['t_max', 't_min']:
                if pd.notna(csv_val) and csv_val != 0.0:
                    weather_data_dict_for_event[col_name_w] = [csv_val]
                else:
                    weather_data_dict_for_event[col_name_w] = [weather_defaults_live.get(col_name_w, 0.0)]
                    if verbose and (pd.isna(csv_val) or csv_val == 0.0):
                        debug_print(f"  '{col_name_w}' uit CSV is {csv_val}, gebruikt default: {weather_data_dict_for_event[col_name_w][0]}", verbose)
            else: # Voor rain_fall, max_wind (0.0 kan een valide CSV waarde zijn)
                weather_data_dict_for_event[col_name_w] = [csv_val if pd.notna(csv_val) else weather_defaults_live.get(col_name_w, 0.0)]

    single_event_dict_for_df = {
        'event_name': [normalized_event_name_internal],
        'first_event_date_start': [event_date_dt_utc_internal],
        'last_event_date_end': [pd.to_datetime(selected_event_data.get('last_event_date_end', event_date_dt_utc_internal), errors='coerce', utc=True).normalize()],
        'city': [
        selected_event_data.get('city')
        if pd.notna(selected_event_data.get('city'))
        else (
            event_specific_history_df['city'].mode().iloc[0]
            if not event_specific_history_df.empty and 'city' in event_specific_history_df.columns
            else 'unknown'
        )
    ],
        'max_capacity': [selected_event_data.get('max_capacity', np.nan)]
    }
    single_event_dict_for_df.update(weather_data_dict_for_event)
    df_events_for_prediction = pd.DataFrame(single_event_dict_for_df)
    for date_col_name_fcheck in ['first_event_date_start', 'last_event_date_end']:
        if date_col_name_fcheck in df_events_for_prediction.columns:
            df_events_for_prediction[date_col_name_fcheck] = pd.to_datetime(df_events_for_prediction[date_col_name_fcheck], errors='coerce', utc=True)

    today_utc_internal_val = pd.Timestamp.now(tz='UTC').normalize()
    days_until_event_actual = (event_date_dt_utc_internal - today_utc_internal_val).days
    t_for_model_selection_val = max(0, days_until_event_actual)

    try:
        closest_t_to_use = get_closest_trained_model_T(t_for_model_selection_val)
    except ValueError as e_model_val_live: return 0, str(e_model_val_live)
    except Exception as e_model_other_live:
        debug_print(f"Fout bij modelselectie: {e_model_other_live}\n{traceback.format_exc()}", True)
        return 0, f"Fout modelselectie: {e_model_other_live}"
        
    debug_print(f"\n--- Details voor Voorspelling ---", verbose)
    debug_print(f"Event: {selected_event_name}, Datum (UTC): {event_date_dt_utc_internal.strftime('%Y-%m-%d')}", verbose)
    debug_print(f"Dagen tot event: {days_until_event_actual}, T-model: {closest_t_to_use}", verbose)
    debug_print(f"--- Einde Details ---\n", verbose)

    # Filter base_tickets_df_context for the specific event to pass as history
    event_specific_history_df = pd.DataFrame()
    if base_tickets_df_context is not None and not base_tickets_df_context.empty and 'event_name' in base_tickets_df_context.columns:
        normalized_event_name_for_filtering = normalize_text(selected_event_name)
        # It's safer to normalize the column in the DataFrame for matching
        base_tickets_df_context['event_name_normalized_for_filter'] = base_tickets_df_context['event_name'].astype(str).apply(normalize_text)
        event_specific_history_df = base_tickets_df_context[
            base_tickets_df_context['event_name_normalized_for_filter'] == normalized_event_name_for_filtering
        ].copy()
        # Drop the temporary normalized column after filtering
        # base_tickets_df_context.drop(columns=['event_name_normalized_for_filter'], inplace=True, errors='ignore') # Be careful modifying input df
        if not event_specific_history_df.empty:
            debug_print(f"Historische data gevonden voor '{selected_event_name}': {len(event_specific_history_df)} rijen.", verbose)
        else:
            debug_print(f"Geen specifieke historische data gevonden voor '{selected_event_name}' in base_tickets_df_context.", verbose)

    try:
        prediction_output_df_final = predict_tickets(
        df_events=df_events_for_prediction,
        T_model=closest_t_to_use,
        known_cities=known_cities_context,
        line_up_df=current_lineup_df_for_event,
        artists_df=current_artists_df_for_event,
        tickets_history_df=event_specific_history_df,        
        verbose=verbose
    )

        
        if prediction_output_df_final.empty or 'predicted_full_event_tickets' not in prediction_output_df_final.columns:
            msg_err = f"Voorspelling mislukt voor '{selected_event_name}' (leeg resultaat intern)."
            debug_print(msg_err, True); return 0, msg_err
        
        final_prediction_val_int = int(prediction_output_df_final['predicted_full_event_tickets'].iloc[0])
        return final_prediction_val_int, f"Model T={closest_t_to_use} ({days_until_event_actual} dgn). Artiesten: Live/Fallback. Weer: {weather_source_info}."
    except Exception as e_pred_final_live:
        err_trace_live = traceback.format_exc()
        debug_print(f"Fout voorspellingsproces '{selected_event_name}': {e_pred_final_live}\n{err_trace_live}", True)
        return 0, f"Fout voorspelling: {e_pred_final_live}"




# ===============================================================
# Hoofdfunctie voor Interactieve Gebruikerstool
# ===============================================================
def interactive_prediction_tool():
    """Interactieve command-line tool voor het maken van ticketvoorspellingen."""
    print("\n" + "="*70)
    print("Capuchin Ticket Voorspeller".center(70))
    print("="*70)
    
    print("\nStap 1: Modellen laden...")
    if not find_and_load_models(verbose=True):
        print("KRITIEK: Geen modellen geladen. Tool stopt.")
        return
    
    print("\nStap 2: Basis datasets laden...")
    base_tickets_df, _, base_artists_fb_df = load_data() 
    
    if base_tickets_df.empty:
        print(f"KRITIEK: Geen ticketdata geladen van '{DEFAULT_TICKETS_PATH}'. Tool stopt.")
        return
    if base_artists_fb_df.empty:
        print(f"WAARSCHUWING: Geen fallback artiestendata ('{DEFAULT_ARTISTS_PATH}') geladen.")
    
    known_cities_for_tool = get_real_cities_present_in_data(base_tickets_df, verbose=True)
    if not known_cities_for_tool:
        print("WAARSCHUWING: Geen bekende steden gevonden in de basisdata. Stadsfeatures zijn mogelijk beperkt.")

    print("\nStap 3: Toekomstige evenementen ophalen...")
    future_events_df_tool = get_future_events(base_tickets_df.copy(), verbose=True) 
    
    if future_events_df_tool.empty:
        print("Geen toekomstige evenementen gevonden om te voorspellen.")
        return
    print(f"\nGevonden: {len(future_events_df_tool)} toekomstige evenementen:")
    print("-"*70)
    events_select_list_tool = []
    for i, (_, event_r_series) in enumerate(future_events_df_tool.iterrows()):
        ev_name_disp = event_r_series.get('event_name', f'Event #{i+1}')
        ev_date_dt_disp = pd.to_datetime(event_r_series.get('first_event_date_start'), errors='coerce', utc=True)
        ev_date_str_disp = ev_date_dt_disp.strftime('%d-%m-%Y (%A)') if pd.notna(ev_date_dt_disp) else "Datum Onbekend"
        ev_days_left_disp = event_r_series.get('days_until_event', 'N/A')
        events_select_list_tool.append({"name": ev_name_disp, "data_row": event_r_series, 
                                     "date_str": ev_date_str_disp, "days_left": ev_days_left_disp})
        print(f"{i+1}. {ev_name_disp} - Datum: {ev_date_str_disp} - Over {ev_days_left_disp} dagen")
    
    while True:
        try:
            print("\nSelecteer een evenement (nr), of 0 om te stoppen:")
            choice_str_input = input("Keuze: ").strip()
            if not choice_str_input.isdigit(): print("Ongeldige invoer."); continue
            choice_idx_input = int(choice_str_input)
            
            if choice_idx_input == 0: print("\nTot ziens!"); break
            if not (1 <= choice_idx_input <= len(events_select_list_tool)):
                print("Ongeldige keuze."); continue

            selected_event_info_dict = events_select_list_tool[choice_idx_input - 1]
            s_ev_name = selected_event_info_dict["name"]
            s_ev_data_series = selected_event_info_dict["data_row"]
            s_ev_date_str = selected_event_info_dict["date_str"]
            s_ev_days_left = selected_event_info_dict["days_left"]
                
            print(f"\nVoorspelling voor: {s_ev_name} ({s_ev_date_str}, over {s_ev_days_left} dgn)")
            verbose_flag_tool = input("Gedetailleerde output? (ja/nee): ").strip().lower().startswith('j')
            manual_lineup_str_input = input("Artiesten (komma-gescheiden, leeg = geen): ").strip()
                                
            print("\nBezig met voorspellen...")
            prediction_val, info_msg_str = predict_event_tickets_live(
                s_ev_name, s_ev_data_series, 
                manual_lineup_str_input if manual_lineup_str_input else None,
                base_tickets_df, base_artists_fb_df, known_cities_for_tool, verbose_flag_tool )
                
            print("\n" + "="*60); print(f"RESULTAAT: {s_ev_name}")
            print(f"  Datum: {s_ev_date_str} ({s_ev_days_left} dgn)")
            print(f"  Voorspelde ticketverkoop: {prediction_val}")
            print(f"  Info: {info_msg_str}")
            if manual_lineup_str_input:
                print("\n  Opgegeven line-up:"); 
                for i, art_disp in enumerate(manual_lineup_str_input.split(','),1): print(f"    {i}. {art_disp.strip()}")
            print("="*60)
        except ValueError: print("Ongeldige numerieke invoer.")
        except Exception as e_tool_loop:
            print(f"!!! FOUT in tool loop: {e_tool_loop}"); print(traceback.format_exc())
        
        if not input("\nNog een voorspelling? (ja/nee): ").strip().lower().startswith('j'):
            print("\nTot ziens!"); break

# ===============================================================
# Script Uitvoering (Entry Point)
# ===============================================================
if __name__ == "__main__":
    interactive_prediction_tool()
