"""
Feature Engineering Module for NFL Big Data Bowl 2026

This module contains all feature engineering functions for trajectory prediction:
- Geometric endpoint computation
- Opponent interaction features
- Route pattern clustering
- GNN-lite neighbor embeddings
- Full feature pipeline with sequence preparation
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# Add project root to sys.path for Config import
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from configs.Config import Config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_velocity(speed, direction_deg):
    """Convert speed and direction (degrees) to velocity components (x, y)."""
    theta = np.deg2rad(direction_deg)
    return speed * np.sin(theta), speed * np.cos(theta)


def height_to_feet(height_str):
    """Convert height string (e.g., '6-2') to feet (e.g., 6.167)."""
    try:
        ft, inches = map(int, str(height_str).split('-'))
        return ft + inches / 12
    except:
        return 6.0


# ============================================================================
# GEOMETRIC FEATURE FUNCTIONS
# ============================================================================

def compute_geometric_endpoint(df):
    """
    Compute where each player SHOULD end up based on geometry.
    This is the deterministic part - no learning needed.
    
    Rules:
    1. Targeted Receivers converge to ball landing position
    2. Defenders mirror receivers (maintain offset)
    3. Others follow momentum (velocity * time)
    
    Args:
        df: DataFrame with player positions, velocities, roles, ball info
        
    Returns:
        DataFrame with added geo_endpoint_x, geo_endpoint_y columns
    """
    df = df.copy()
    
    # Time to play end
    if 'num_frames_output' in df.columns:
        t_total = df['num_frames_output'] / 10.0
    else:
        t_total = 3.0
    
    df['time_to_endpoint'] = t_total
    
    # Initialize with momentum (default rule)
    df['geo_endpoint_x'] = df['x'] + df['velocity_x'] * t_total
    df['geo_endpoint_y'] = df['y'] + df['velocity_y'] * t_total
    
    # Rule 1: Targeted Receivers converge to ball
    if 'ball_land_x' in df.columns:
        receiver_mask = df['player_role'] == 'Targeted Receiver'
        df.loc[receiver_mask, 'geo_endpoint_x'] = df.loc[receiver_mask, 'ball_land_x']
        df.loc[receiver_mask, 'geo_endpoint_y'] = df.loc[receiver_mask, 'ball_land_y']
        
        # Rule 2: Defenders mirror receivers (maintain offset)
        defender_mask = df['player_role'] == 'Defensive Coverage'
        has_mirror = df.get('mirror_offset_x', 0).notna() & (df.get('mirror_wr_dist', 50) < 15)
        coverage_mask = defender_mask & has_mirror
        
        df.loc[coverage_mask, 'geo_endpoint_x'] = (
            df.loc[coverage_mask, 'ball_land_x'] + 
            df.loc[coverage_mask, 'mirror_offset_x'].fillna(0)
        )
        df.loc[coverage_mask, 'geo_endpoint_y'] = (
            df.loc[coverage_mask, 'ball_land_y'] + 
            df.loc[coverage_mask, 'mirror_offset_y'].fillna(0)
        )
    
    # Clip to field boundaries
    df['geo_endpoint_x'] = df['geo_endpoint_x'].clip(Config.FIELD_X_MIN, Config.FIELD_X_MAX)
    df['geo_endpoint_y'] = df['geo_endpoint_y'].clip(Config.FIELD_Y_MIN, Config.FIELD_Y_MAX)
    
    return df


def add_geometric_features(df):
    """
    Add features that describe the geometric solution.
    
    Computes:
    - Vector to geometric endpoint
    - Required velocity/acceleration to reach endpoint
    - Current velocity error vs required
    - Alignment with geometric path
    
    Args:
        df: DataFrame (will be modified in-place via copy)
        
    Returns:
        DataFrame with added geometric feature columns
    """
    df = compute_geometric_endpoint(df)
    
    # Vector to geometric endpoint
    df['geo_vector_x'] = df['geo_endpoint_x'] - df['x']
    df['geo_vector_y'] = df['geo_endpoint_y'] - df['y']
    df['geo_distance'] = np.sqrt(df['geo_vector_x']**2 + df['geo_vector_y']**2)
    
    # Required velocity to reach geometric endpoint
    t = df['time_to_endpoint'] + 0.1
    df['geo_required_vx'] = df['geo_vector_x'] / t
    df['geo_required_vy'] = df['geo_vector_y'] / t
    
    # Current velocity vs required
    df['geo_velocity_error_x'] = df['geo_required_vx'] - df['velocity_x']
    df['geo_velocity_error_y'] = df['geo_required_vy'] - df['velocity_y']
    df['geo_velocity_error'] = np.sqrt(
        df['geo_velocity_error_x']**2 + df['geo_velocity_error_y']**2
    )
    
    # Required constant acceleration (a = 2*Δx/t²)
    t_sq = t * t
    df['geo_required_ax'] = 2 * df['geo_vector_x'] / t_sq
    df['geo_required_ay'] = 2 * df['geo_vector_y'] / t_sq
    df['geo_required_ax'] = df['geo_required_ax'].clip(-10, 10)
    df['geo_required_ay'] = df['geo_required_ay'].clip(-10, 10)
    
    # Alignment with geometric path
    velocity_mag = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    geo_unit_x = df['geo_vector_x'] / (df['geo_distance'] + 0.1)
    geo_unit_y = df['geo_vector_y'] / (df['geo_distance'] + 0.1)
    df['geo_alignment'] = (
        df['velocity_x'] * geo_unit_x + df['velocity_y'] * geo_unit_y
    ) / (velocity_mag + 0.1)
    
    # Role-specific geometric quality
    df['geo_receiver_urgency'] = df['is_receiver'] * df['geo_distance'] / (t + 0.1)
    df['geo_defender_coupling'] = df['is_coverage'] * (1.0 / (df.get('mirror_wr_dist', 50) + 1.0))
    
    return df


# ============================================================================
# OPPONENT INTERACTION FEATURES
# ============================================================================

def get_opponent_features(input_df):
    """
    Compute opponent interaction features with MIRROR WR tracking.
    
    Features:
    - Nearest opponent distance
    - Closing speed
    - Number of nearby opponents
    - Mirror features for defensive coverage (matching receiver movement)
    
    Args:
        input_df: DataFrame with game_id, play_id, nfl_id, positions, velocities
        
    Returns:
        DataFrame with opponent features per (game_id, play_id, nfl_id)
    """
    features = []
    
    for (gid, pid), group in tqdm(input_df.groupby(['game_id', 'play_id']), 
                                   desc="🏈 Opponents", leave=False):
        last = group.sort_values('frame_id').groupby('nfl_id').last()
        
        if len(last) < 2:
            continue
            
        positions = last[['x', 'y']].values
        sides = last['player_side'].values
        speeds = last['s'].values
        directions = last['dir'].values
        roles = last['player_role'].values
        
        receiver_mask = np.isin(roles, ['Targeted Receiver', 'Other Route Runner'])
        
        for i, (nid, side, role) in enumerate(zip(last.index, sides, roles)):
            opp_mask = sides != side
            
            feat = {
                'game_id': gid, 'play_id': pid, 'nfl_id': nid,
                'nearest_opp_dist': 50.0, 'closing_speed': 0.0,
                'num_nearby_opp_3': 0, 'num_nearby_opp_5': 0,
                'mirror_wr_vx': 0.0, 'mirror_wr_vy': 0.0,
                'mirror_offset_x': 0.0, 'mirror_offset_y': 0.0,
                'mirror_wr_dist': 50.0,
            }
            
            if not opp_mask.any():
                features.append(feat)
                continue
            
            opp_positions = positions[opp_mask]
            distances = np.sqrt(((positions[i] - opp_positions)**2).sum(axis=1))
            
            if len(distances) == 0:
                features.append(feat)
                continue
                
            nearest_idx = distances.argmin()
            feat['nearest_opp_dist'] = distances[nearest_idx]
            feat['num_nearby_opp_3'] = (distances < 3.0).sum()
            feat['num_nearby_opp_5'] = (distances < 5.0).sum()
            
            my_vx, my_vy = get_velocity(speeds[i], directions[i])
            opp_speeds = speeds[opp_mask]
            opp_dirs = directions[opp_mask]
            opp_vx, opp_vy = get_velocity(opp_speeds[nearest_idx], opp_dirs[nearest_idx])
            
            rel_vx = my_vx - opp_vx
            rel_vy = my_vy - opp_vy
            to_me = positions[i] - opp_positions[nearest_idx]
            to_me_norm = to_me / (np.linalg.norm(to_me) + 0.1)
            feat['closing_speed'] = -(rel_vx * to_me_norm[0] + rel_vy * to_me_norm[1])
            
            # Mirror WR tracking for defensive coverage
            if role == 'Defensive Coverage' and receiver_mask.any():
                rec_positions = positions[receiver_mask]
                rec_distances = np.sqrt(((positions[i] - rec_positions)**2).sum(axis=1))
                
                if len(rec_distances) > 0:
                    closest_rec_idx = rec_distances.argmin()
                    rec_indices = np.where(receiver_mask)[0]
                    actual_rec_idx = rec_indices[closest_rec_idx]
                    
                    rec_vx, rec_vy = get_velocity(speeds[actual_rec_idx], directions[actual_rec_idx])
                    
                    feat['mirror_wr_vx'] = rec_vx
                    feat['mirror_wr_vy'] = rec_vy
                    feat['mirror_wr_dist'] = rec_distances[closest_rec_idx]
                    feat['mirror_offset_x'] = positions[i][0] - rec_positions[closest_rec_idx][0]
                    feat['mirror_offset_y'] = positions[i][1] - rec_positions[closest_rec_idx][1]
            
            features.append(feat)
    
    return pd.DataFrame(features)


# ============================================================================
# ROUTE PATTERN CLUSTERING
# ============================================================================

def extract_route_patterns(input_df, kmeans=None, scaler=None, fit=True):
    """
    Extract and cluster route patterns from player trajectories.
    
    Features extracted:
    - Trajectory straightness
    - Turn angles (max, mean)
    - Depth/width of route
    - Speed characteristics
    
    Args:
        input_df: DataFrame with player trajectories
        kmeans: Pre-fitted KMeans model (if fit=False)
        scaler: Pre-fitted StandardScaler (if fit=False)
        fit: If True, fit new models; if False, use provided models
        
    Returns:
        If fit=True: (route_df, kmeans, scaler)
        If fit=False: route_df (with route_pattern column)
    """
    route_features = []
    
    for (gid, pid, nid), group in tqdm(input_df.groupby(['game_id', 'play_id', 'nfl_id']), 
                                        desc="🛣️  Routes", leave=False):
        traj = group.sort_values('frame_id').tail(5)
        
        if len(traj) < 3:
            continue
        
        positions = traj[['x', 'y']].values
        speeds = traj['s'].values
        
        total_dist = np.sum(np.sqrt(np.diff(positions[:, 0])**2 + np.diff(positions[:, 1])**2))
        displacement = np.sqrt((positions[-1, 0] - positions[0, 0])**2 + 
                              (positions[-1, 1] - positions[0, 1])**2)
        straightness = displacement / (total_dist + 0.1)
        
        angles = np.arctan2(np.diff(positions[:, 1]), np.diff(positions[:, 0]))
        if len(angles) > 1:
            angle_changes = np.abs(np.diff(angles))
            max_turn = np.max(angle_changes)
            mean_turn = np.mean(angle_changes)
        else:
            max_turn = mean_turn = 0
        
        speed_mean = speeds.mean()
        speed_change = speeds[-1] - speeds[0] if len(speeds) > 1 else 0
        dx = positions[-1, 0] - positions[0, 0]
        dy = positions[-1, 1] - positions[0, 1]
        
        route_features.append({
            'game_id': gid, 'play_id': pid, 'nfl_id': nid,
            'traj_straightness': straightness,
            'traj_max_turn': max_turn,
            'traj_mean_turn': mean_turn,
            'traj_depth': abs(dx),
            'traj_width': abs(dy),
            'speed_mean': speed_mean,
            'speed_change': speed_change,
        })
    
    route_df = pd.DataFrame(route_features)
    feat_cols = ['traj_straightness', 'traj_max_turn', 'traj_mean_turn',
                 'traj_depth', 'traj_width', 'speed_mean', 'speed_change']
    X = route_df[feat_cols].fillna(0)
    
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=Config.N_ROUTE_CLUSTERS, random_state=Config.SEED, n_init=10)
        route_df['route_pattern'] = kmeans.fit_predict(X_scaled)
        return route_df, kmeans, scaler
    else:
        X_scaled = scaler.transform(X)
        route_df['route_pattern'] = kmeans.predict(X_scaled)
        return route_df


# ============================================================================
# GNN-LITE NEIGHBOR EMBEDDINGS
# ============================================================================

def compute_neighbor_embeddings(input_df, k_neigh=Config.K_NEIGH, 
                                radius=Config.RADIUS, tau=Config.TAU):
    """
    Compute GNN-lite embeddings by aggregating nearby player information.
    
    Creates weighted aggregations of neighbors' positions and velocities,
    distinguishing between allies and opponents. Weights decay exponentially with distance.
    
    Args:
        input_df: DataFrame with player positions, velocities, sides
        k_neigh: Maximum number of neighbors to consider
        radius: Maximum distance to consider neighbors
        tau: Exponential decay parameter for distance weighting
        
    Returns:
        DataFrame with GNN embedding features per (game_id, play_id, nfl_id)
    """
    print("🕸️  GNN embeddings...")
    
    cols_needed = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", 
                   "velocity_x", "velocity_y", "player_side"]
    src = input_df[cols_needed].copy()
    
    # Get last frame for each player
    last = (src.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
               .groupby(["game_id", "play_id", "nfl_id"], as_index=False)
               .tail(1)
               .rename(columns={"frame_id": "last_frame_id"})
               .reset_index(drop=True))
    
    # Merge with all players in same frame to find neighbors
    tmp = last.merge(
        src.rename(columns={
            "frame_id": "nb_frame_id", "nfl_id": "nfl_id_nb",
            "x": "x_nb", "y": "y_nb", 
            "velocity_x": "vx_nb", "velocity_y": "vy_nb", 
            "player_side": "player_side_nb"
        }),
        left_on=["game_id", "play_id", "last_frame_id"],
        right_on=["game_id", "play_id", "nb_frame_id"],
        how="left"
    )
    
    # Remove self
    tmp = tmp[tmp["nfl_id_nb"] != tmp["nfl_id"]]
    
    # Compute differences and distances
    tmp["dx"] = tmp["x_nb"] - tmp["x"]
    tmp["dy"] = tmp["y_nb"] - tmp["y"]
    tmp["dvx"] = tmp["vx_nb"] - tmp["velocity_x"]
    tmp["dvy"] = tmp["vy_nb"] - tmp["velocity_y"]
    tmp["dist"] = np.sqrt(tmp["dx"]**2 + tmp["dy"]**2)
    
    # Filter valid neighbors
    tmp = tmp[np.isfinite(tmp["dist"]) & (tmp["dist"] > 1e-6)]
    if radius is not None:
        tmp = tmp[tmp["dist"] <= radius]
    
    # Identify allies vs opponents
    tmp["is_ally"] = (tmp["player_side_nb"] == tmp["player_side"]).astype(np.float32)
    
    # Rank neighbors by distance and keep top k
    keys = ["game_id", "play_id", "nfl_id"]
    tmp["rnk"] = tmp.groupby(keys)["dist"].rank(method="first")
    if k_neigh is not None:
        tmp = tmp[tmp["rnk"] <= float(k_neigh)]
    
    # Exponential distance weighting
    tmp["w"] = np.exp(-tmp["dist"] / float(tau))
    sum_w = tmp.groupby(keys)["w"].transform("sum")
    tmp["wn"] = np.where(sum_w > 0, tmp["w"] / sum_w, 0.0)
    
    # Separate weights for allies and opponents
    tmp["wn_ally"] = tmp["wn"] * tmp["is_ally"]
    tmp["wn_opp"] = tmp["wn"] * (1.0 - tmp["is_ally"])
    
    # Weighted differences
    for col in ["dx", "dy", "dvx", "dvy"]:
        tmp[f"{col}_ally_w"] = tmp[col] * tmp["wn_ally"]
        tmp[f"{col}_opp_w"] = tmp[col] * tmp["wn_opp"]
    
    # Distance statistics
    tmp["dist_ally"] = np.where(tmp["is_ally"] > 0.5, tmp["dist"], np.nan)
    tmp["dist_opp"] = np.where(tmp["is_ally"] < 0.5, tmp["dist"], np.nan)
    
    # Aggregate weighted features
    ag = tmp.groupby(keys).agg(
        gnn_ally_dx_mean=("dx_ally_w", "sum"),
        gnn_ally_dy_mean=("dy_ally_w", "sum"),
        gnn_ally_dvx_mean=("dvx_ally_w", "sum"),
        gnn_ally_dvy_mean=("dvy_ally_w", "sum"),
        gnn_opp_dx_mean=("dx_opp_w", "sum"),
        gnn_opp_dy_mean=("dy_opp_w", "sum"),
        gnn_opp_dvx_mean=("dvx_opp_w", "sum"),
        gnn_opp_dvy_mean=("dvy_opp_w", "sum"),
        gnn_ally_cnt=("is_ally", "sum"),
        gnn_opp_cnt=("is_ally", lambda s: float(len(s) - s.sum())),
        gnn_ally_dmin=("dist_ally", "min"),
        gnn_ally_dmean=("dist_ally", "mean"),
        gnn_opp_dmin=("dist_opp", "min"),
        gnn_opp_dmean=("dist_opp", "mean"),
    ).reset_index()
    
    # Add distance to nearest 3 neighbors
    near = tmp.loc[tmp["rnk"] <= 3, keys + ["rnk", "dist"]].copy()
    if len(near) > 0:
        near["rnk"] = near["rnk"].astype(int)
        dwide = near.pivot_table(index=keys, columns="rnk", values="dist", aggfunc="first")
        dwide = dwide.rename(columns={1: "gnn_d1", 2: "gnn_d2", 3: "gnn_d3"}).reset_index()
        ag = ag.merge(dwide, on=keys, how="left")
    
    # Fill missing values
    for c in ["gnn_ally_dx_mean", "gnn_ally_dy_mean", "gnn_ally_dvx_mean", "gnn_ally_dvy_mean",
              "gnn_opp_dx_mean", "gnn_opp_dy_mean", "gnn_opp_dvx_mean", "gnn_opp_dvy_mean"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_cnt", "gnn_opp_cnt"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_dmin", "gnn_opp_dmin", "gnn_ally_dmean", "gnn_opp_dmean", 
              "gnn_d1", "gnn_d2", "gnn_d3"]:
        ag[c] = ag[c].fillna(radius if radius is not None else 30.0)
    
    return ag


# ============================================================================
# MAIN FEATURE PIPELINE
# ============================================================================

def prepare_sequences_geometric(input_df, output_df=None, test_template=None, 
                                is_training=True, window_size=10,
                                route_kmeans=None, route_scaler=None):
    """
    Main feature engineering pipeline that prepares sequences for model training/inference.
    
    Steps:
    1. Base features (velocity, acceleration, role indicators, ball relationships)
    2. Advanced features (opponent interactions, route patterns, GNN embeddings)
    3. Temporal features (lags, rolling stats, EMA)
    4. Time-based features (play progress, time remaining)
    5. Geometric features (endpoint prediction, required velocity/acceleration)
    6. Sequence creation (fixed-size windows for model input)
    
    Args:
        input_df: Input DataFrame with raw player tracking data
        output_df: Output DataFrame with ground truth (required if is_training=True)
        test_template: Template DataFrame with target frame_ids (required if is_training=False)
        is_training: Whether preparing for training or inference
        window_size: Number of frames in input sequence
        route_kmeans: Pre-fitted route clustering model (for inference)
        route_scaler: Pre-fitted route scaler (for inference)
        
    Returns:
        If is_training=True:
            (sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids,
             geo_endpoints_x, geo_endpoints_y, route_kmeans, route_scaler)
        If is_training=False:
            (sequences, sequence_ids, geo_endpoints_x, geo_endpoints_y)
    """
    print(f"\n{'='*80}")
    print(f"PREPARING GEOMETRIC SEQUENCES")
    print(f"{'='*80}")
    
    input_df = input_df.copy()
    input_df = input_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    # ========================================================================
    # STEP 1: BASE FEATURES
    # ========================================================================
    print("Step 1: Base features...")
    
    # Physical attributes
    input_df['player_height_feet'] = input_df['player_height'].apply(height_to_feet)
    height_parts = input_df['player_height'].str.split('-', expand=True)
    input_df['height_inches'] = height_parts[0].astype(float) * 12 + height_parts[1].astype(float)
    input_df['bmi'] = (input_df['player_weight'] / (input_df['height_inches']**2)) * 703
    
    # Velocity components from speed and direction
    dir_rad = np.deg2rad(input_df['dir'].fillna(0))
    input_df['velocity_x'] = input_df['s'] * np.sin(dir_rad)
    input_df['velocity_y'] = input_df['s'] * np.cos(dir_rad)
    input_df['acceleration_x'] = input_df['a'] * np.cos(dir_rad)
    input_df['acceleration_y'] = input_df['a'] * np.sin(dir_rad)
    
    # Physics features
    input_df['speed_squared'] = input_df['s'] ** 2
    input_df['accel_magnitude'] = np.sqrt(input_df['acceleration_x']**2 + input_df['acceleration_y']**2)
    input_df['momentum_x'] = input_df['velocity_x'] * input_df['player_weight']
    input_df['momentum_y'] = input_df['velocity_y'] * input_df['player_weight']
    input_df['kinetic_energy'] = 0.5 * input_df['player_weight'] * input_df['speed_squared']
    
    # Orientation
    input_df['orientation_diff'] = np.abs(input_df['o'] - input_df['dir'])
    input_df['orientation_diff'] = np.minimum(input_df['orientation_diff'], 360 - input_df['orientation_diff'])
    
    # Role indicators
    input_df['is_offense'] = (input_df['player_side'] == 'Offense').astype(int)
    input_df['is_defense'] = (input_df['player_side'] == 'Defense').astype(int)
    input_df['is_receiver'] = (input_df['player_role'] == 'Targeted Receiver').astype(int)
    input_df['is_coverage'] = (input_df['player_role'] == 'Defensive Coverage').astype(int)
    input_df['is_passer'] = (input_df['player_role'] == 'Passer').astype(int)
    input_df['role_targeted_receiver'] = input_df['is_receiver']
    input_df['role_defensive_coverage'] = input_df['is_coverage']
    input_df['role_passer'] = input_df['is_passer']
    input_df['side_offense'] = input_df['is_offense']
    
    # Ball relationships
    if 'ball_land_x' in input_df.columns:
        ball_dx = input_df['ball_land_x'] - input_df['x']
        ball_dy = input_df['ball_land_y'] - input_df['y']
        input_df['distance_to_ball'] = np.sqrt(ball_dx**2 + ball_dy**2)
        input_df['dist_to_ball'] = input_df['distance_to_ball']
        input_df['dist_squared'] = input_df['distance_to_ball'] ** 2
        input_df['angle_to_ball'] = np.arctan2(ball_dy, ball_dx)
        input_df['ball_direction_x'] = ball_dx / (input_df['distance_to_ball'] + 1e-6)
        input_df['ball_direction_y'] = ball_dy / (input_df['distance_to_ball'] + 1e-6)
        input_df['closing_speed_ball'] = (
            input_df['velocity_x'] * input_df['ball_direction_x'] +
            input_df['velocity_y'] * input_df['ball_direction_y']
        )
        input_df['velocity_toward_ball'] = (
            input_df['velocity_x'] * np.cos(input_df['angle_to_ball']) + 
            input_df['velocity_y'] * np.sin(input_df['angle_to_ball'])
        )
        input_df['velocity_alignment'] = np.cos(input_df['angle_to_ball'] - dir_rad)
        input_df['angle_diff'] = np.abs(input_df['o'] - np.degrees(input_df['angle_to_ball']))
        input_df['angle_diff'] = np.minimum(input_df['angle_diff'], 360 - input_df['angle_diff'])
    
    # ========================================================================
    # STEP 2: ADVANCED FEATURES
    # ========================================================================
    print("Step 2: Advanced features...")
    
    # Opponent interactions
    opp_features = get_opponent_features(input_df)
    input_df = input_df.merge(opp_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    # Route patterns
    if is_training:
        route_features, route_kmeans, route_scaler = extract_route_patterns(input_df)
    else:
        route_features = extract_route_patterns(input_df, route_kmeans, route_scaler, fit=False)
    input_df = input_df.merge(route_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    # GNN-lite neighbor embeddings
    gnn_features = compute_neighbor_embeddings(input_df)
    input_df = input_df.merge(gnn_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    # Pressure features
    if 'nearest_opp_dist' in input_df.columns:
        input_df['pressure'] = 1 / np.maximum(input_df['nearest_opp_dist'], 0.5)
        input_df['under_pressure'] = (input_df['nearest_opp_dist'] < 3).astype(int)
        input_df['pressure_x_speed'] = input_df['pressure'] * input_df['s']
    
    # Mirror features
    if 'mirror_wr_vx' in input_df.columns:
        s_safe = np.maximum(input_df['s'], 0.1)
        input_df['mirror_similarity'] = (
            input_df['velocity_x'] * input_df['mirror_wr_vx'] + 
            input_df['velocity_y'] * input_df['mirror_wr_vy']
        ) / s_safe
        input_df['mirror_offset_dist'] = np.sqrt(
            input_df['mirror_offset_x']**2 + input_df['mirror_offset_y']**2
        )
        input_df['mirror_alignment'] = input_df['mirror_similarity'] * input_df['role_defensive_coverage']
    
    # ========================================================================
    # STEP 3: TEMPORAL FEATURES
    # ========================================================================
    print("Step 3: Temporal features...")
    
    gcols = ['game_id', 'play_id', 'nfl_id']
    
    # Lag features
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            if col in input_df.columns:
                input_df[f'{col}_lag{lag}'] = input_df.groupby(gcols)[col].shift(lag)
    
    # Rolling statistics
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            if col in input_df.columns:
                input_df[f'{col}_rolling_mean_{window}'] = (
                    input_df.groupby(gcols)[col]
                      .rolling(window, min_periods=1).mean()
                      .reset_index(level=[0,1,2], drop=True)
                )
                input_df[f'{col}_rolling_std_{window}'] = (
                    input_df.groupby(gcols)[col]
                      .rolling(window, min_periods=1).std()
                      .reset_index(level=[0,1,2], drop=True)
                )
    
    # Velocity deltas
    for col in ['velocity_x', 'velocity_y']:
        if col in input_df.columns:
            input_df[f'{col}_delta'] = input_df.groupby(gcols)[col].diff()
    
    # Exponential moving averages
    input_df['velocity_x_ema'] = input_df.groupby(gcols)['velocity_x'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    input_df['velocity_y_ema'] = input_df.groupby(gcols)['velocity_y'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    input_df['speed_ema'] = input_df.groupby(gcols)['s'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    
    # ========================================================================
    # STEP 4: TIME-BASED FEATURES
    # ========================================================================
    print("Step 4: Time features...")
    
    if 'num_frames_output' in input_df.columns:
        max_frames = input_df['num_frames_output']
        
        input_df['max_play_duration'] = max_frames / 10.0
        input_df['frame_time'] = input_df['frame_id'] / 10.0
        input_df['progress_ratio'] = input_df['frame_id'] / np.maximum(max_frames, 1)
        input_df['time_remaining'] = (max_frames - input_df['frame_id']) / 10.0
        input_df['frames_remaining'] = max_frames - input_df['frame_id']
        
        input_df['expected_x_at_ball'] = input_df['x'] + input_df['velocity_x'] * input_df['frame_time']
        input_df['expected_y_at_ball'] = input_df['y'] + input_df['velocity_y'] * input_df['frame_time']
        
        if 'ball_land_x' in input_df.columns:
            input_df['error_from_ball_x'] = input_df['expected_x_at_ball'] - input_df['ball_land_x']
            input_df['error_from_ball_y'] = input_df['expected_y_at_ball'] - input_df['ball_land_y']
            input_df['error_from_ball'] = np.sqrt(
                input_df['error_from_ball_x']**2 + input_df['error_from_ball_y']**2
            )
            
            input_df['weighted_dist_by_time'] = input_df['dist_to_ball'] / (input_df['frame_time'] + 0.1)
            input_df['dist_scaled_by_progress'] = input_df['dist_to_ball'] * (1 - input_df['progress_ratio'])
        
        input_df['time_squared'] = input_df['frame_time'] ** 2
        input_df['velocity_x_progress'] = input_df['velocity_x'] * input_df['progress_ratio']
        input_df['velocity_y_progress'] = input_df['velocity_y'] * input_df['progress_ratio']
        input_df['speed_scaled_by_time_left'] = input_df['s'] * input_df['time_remaining']
        
        input_df['actual_play_length'] = max_frames
        input_df['length_ratio'] = max_frames / 30.0
    
    # ========================================================================
    # STEP 5: GEOMETRIC FEATURES
    # ========================================================================
    print("Step 5: 🎯 Geometric endpoint features...")
    input_df = add_geometric_features(input_df)
    
    # ========================================================================
    # STEP 6: BUILD FEATURE LIST
    # ========================================================================
    print("Step 6: Building feature list...")
    
    # Base features
    feature_cols = [
        'x', 'y', 's', 'a', 'o', 'dir', 'frame_id', 'ball_land_x', 'ball_land_y',
        'player_height_feet', 'player_weight', 'height_inches', 'bmi',
        'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',
        'momentum_x', 'momentum_y', 'kinetic_energy',
        'speed_squared', 'accel_magnitude', 'orientation_diff',
        'is_offense', 'is_defense', 'is_receiver', 'is_coverage', 'is_passer',
        'role_targeted_receiver', 'role_defensive_coverage', 'role_passer', 'side_offense',
        'distance_to_ball', 'dist_to_ball', 'dist_squared', 'angle_to_ball', 
        'ball_direction_x', 'ball_direction_y', 'closing_speed_ball',
        'velocity_toward_ball', 'velocity_alignment', 'angle_diff',
        'nearest_opp_dist', 'closing_speed', 'num_nearby_opp_3', 'num_nearby_opp_5',
        'mirror_wr_vx', 'mirror_wr_vy', 'mirror_offset_x', 'mirror_offset_y',
        'pressure', 'under_pressure', 'pressure_x_speed', 
        'mirror_similarity', 'mirror_offset_dist', 'mirror_alignment',
        'route_pattern', 'traj_straightness', 'traj_max_turn', 'traj_mean_turn',
        'traj_depth', 'traj_width', 'speed_mean', 'speed_change',
        'gnn_ally_dx_mean', 'gnn_ally_dy_mean', 'gnn_ally_dvx_mean', 'gnn_ally_dvy_mean',
        'gnn_opp_dx_mean', 'gnn_opp_dy_mean', 'gnn_opp_dvx_mean', 'gnn_opp_dvy_mean',
        'gnn_ally_cnt', 'gnn_opp_cnt',
        'gnn_ally_dmin', 'gnn_ally_dmean', 'gnn_opp_dmin', 'gnn_opp_dmean',
        'gnn_d1', 'gnn_d2', 'gnn_d3',
    ]
    
    # Lag features
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            feature_cols.append(f'{col}_lag{lag}')
    
    # Rolling features
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            feature_cols.append(f'{col}_rolling_mean_{window}')
            feature_cols.append(f'{col}_rolling_std_{window}')
    
    # Delta and EMA features
    feature_cols.extend(['velocity_x_delta', 'velocity_y_delta'])
    feature_cols.extend(['velocity_x_ema', 'velocity_y_ema', 'speed_ema'])
    
    # Time features
    feature_cols.extend([
        'max_play_duration', 'frame_time', 'progress_ratio', 'time_remaining', 'frames_remaining',
        'expected_x_at_ball', 'expected_y_at_ball', 
        'error_from_ball_x', 'error_from_ball_y', 'error_from_ball',
        'time_squared', 'weighted_dist_by_time', 
        'velocity_x_progress', 'velocity_y_progress', 'dist_scaled_by_progress',
        'speed_scaled_by_time_left', 'actual_play_length', 'length_ratio',
    ])
    
    # Geometric features
    feature_cols.extend([
        'geo_endpoint_x', 'geo_endpoint_y',
        'geo_vector_x', 'geo_vector_y', 'geo_distance',
        'geo_required_vx', 'geo_required_vy',
        'geo_velocity_error_x', 'geo_velocity_error_y', 'geo_velocity_error',
        'geo_required_ax', 'geo_required_ay',
        'geo_alignment',
    ])
    
    # Filter to only existing columns
    feature_cols = [c for c in feature_cols if c in input_df.columns]
    print(f"✓ Using {len(feature_cols)} features (154 proven + 13 geometric)")
    
    # ========================================================================
    # STEP 7: CREATE SEQUENCES
    # ========================================================================
    print("Step 7: Creating sequences...")
    
    input_df.set_index(['game_id', 'play_id', 'nfl_id'], inplace=True)
    grouped = input_df.groupby(level=['game_id', 'play_id', 'nfl_id'])
    
    target_rows = output_df if is_training else test_template
    target_groups = target_rows[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids = [], [], [], [], []
    geo_endpoints_x, geo_endpoints_y = [], []
    
    for _, row in tqdm(target_groups.iterrows(), total=len(target_groups), desc="Creating sequences"):
        key = (row['game_id'], row['play_id'], row['nfl_id'])
        
        try:
            group_df = grouped.get_group(key)
        except KeyError:
            continue
        
        # Extract window of last N frames
        input_window = group_df.tail(window_size)
        
        # Pad if needed (for inference)
        if len(input_window) < window_size:
            if is_training:
                continue
            pad_len = window_size - len(input_window)
            pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=input_window.columns)
            input_window = pd.concat([pad_df, input_window], ignore_index=True)
        
        # Fill missing values
        input_window = input_window.fillna(group_df.mean(numeric_only=True))
        seq = input_window[feature_cols].values
        
        # Handle NaN values
        if np.isnan(seq).any():
            if is_training:
                continue
            seq = np.nan_to_num(seq, nan=0.0)
        
        sequences.append(seq)
        
        # Store geometric endpoint for this player
        geo_x = input_window.iloc[-1]['geo_endpoint_x']
        geo_y = input_window.iloc[-1]['geo_endpoint_y']
        geo_endpoints_x.append(geo_x)
        geo_endpoints_y.append(geo_y)
        
        # Compute targets for training
        if is_training:
            out_grp = output_df[
                (output_df['game_id']==row['game_id']) &
                (output_df['play_id']==row['play_id']) &
                (output_df['nfl_id']==row['nfl_id'])
            ].sort_values('frame_id')
            
            last_x = input_window.iloc[-1]['x']
            last_y = input_window.iloc[-1]['y']
            
            dx = out_grp['x'].values - last_x
            dy = out_grp['y'].values - last_y
            
            targets_dx.append(dx)
            targets_dy.append(dy)
            targets_frame_ids.append(out_grp['frame_id'].values)
        
        # Store sequence metadata
        sequence_ids.append({
            'game_id': key[0],
            'play_id': key[1],
            'nfl_id': key[2],
            'frame_id': input_window.iloc[-1]['frame_id']
        })
    
    print(f"✓ Created {len(sequences)} sequences")
    
    # Return appropriate outputs
    if is_training:
        return (sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids, 
                geo_endpoints_x, geo_endpoints_y, route_kmeans, route_scaler)
    return sequences, sequence_ids, geo_endpoints_x, geo_endpoints_y
