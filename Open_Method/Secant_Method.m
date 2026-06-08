function main_beam_deflection_secant_nonlinear()
    % =====================================================================
    % 1. PHYSICAL CONSTANTS & BEAM PROFILE PROPERTIES
    % =====================================================================
    L = 0.5;           % Length of the cantilever beam (meters)
    b = 0.03;          % Width of the beam cross-section (meters)
    t = 0.01;          % Thickness of the beam cross-section (meters)
    P = 1000.0;        % Concentrated point load acting downwards at the free tip (N)
    w_area = 38500;    % Distributed payload pressure acting per unit area (N/m^2)
    E = 200e9;         % Young's Modulus of Elasticity (Structural Steel, 200 GPa)
    I = 2.50e-09;      % Area Moment of Inertia for bending resistance (m^4)
    
    % Transform the 2D area pressure into a 1D line load along the beam length
    q = w_area * b;    % Linear load distribution (N/m)
    EI = E * I;        % Flexural Rigidity constant of the structure
    
    % =====================================================================
    % 2. NUMERICAL GRID MESH DISCRETIZATION
    % =====================================================================
    num_steps = 150;                  % Total spatial nodes across the beam
    s_mesh = linspace(0, L, num_steps); % Coordinate vectors tracking along the arc length (s)
    
    % =====================================================================
    % 3. GEOMETRICALLY NON-LINEAR SYSTEM OF ODEs (State Vector Machine)
    % =====================================================================
    % State Vectors mapping: Y(1) = Vertical Deflection (y), Y(2) = Slope angle (theta), Y(3) = Internal Moment (M/EI)
    % Notice that cos(Y(2)) scales the weight vector as the structural incline steepens.
    f_system = @(s, Y) [Y(2); ...
                        Y(3); ...
                        -(q/EI)*(L - s)*cos(Y(2)) - (P/EI)*cos(Y(2))];
    
    % =====================================================================
    % 4. THE SHOOTING METHOD MECHANICS (SECANT ITERATION)
    % =====================================================================
    % Secant requires two active starting guesses for the unknown wall moment constraint Y(3,1)
    guess1 = 0.0;     % Initial guess 1 (Assuming a flat, unloaded baseline response)
    guess2 = -0.01;   % Initial guess 2 (Slightly negative perturbation starter)
    
    % Execute forward numerical integration for the first historical guess slot
    sol1 = RK4(f_system, s_mesh, [0; 0; guess1]); 
    error1 = sol1(3, end); % Extract the residual error (Target tip moment must equal 0)
    
    % Execute forward numerical integration for the second historical guess slot
    sol2 = RK4(f_system, s_mesh, [0; 0; guess2]); 
    error2 = sol2(3, end); % Extract the residual error
    
    tolerance = 1e-9;      % Accuracy limit threshold for convergence
    max_iter = 50;         % Safety iteration cap
    converged = false;     % Default loop tracking state flag
    
    fprintf('\n--- SECANT METHOD ITERATION LOG (GEOMETRIC NON-LINEAR) ---\n');
    fprintf('%-10s %-15s %-15s %-15s\n', 'Iter', 'Old Guess', 'New Guess', 'Residual Moment Error');
    
    for iter = 1:max_iter
        % Prevent a catastrophic division by zero failure if errors converge exactly
        if abs(error2 - error1) < 1e-15, break; end
        
        % SECANT UPDATE LAW: Project linear extrapolation line using two historical nodes
        next_guess = guess2 - error2 * (guess2 - guess1) / (error2 - error1);
        
        % Test the newly generated guess by integrating across the mesh space
        sol_next = RK4(f_system, s_mesh, [0; 0; next_guess]); 
        error_next = sol_next(3, end); % Sample new residual error at the free boundary tip
        
        % Log execution parameters to monitor convergence behavior
        fprintf('%-10d %-15.4f %-15.4f %-15.4e\n', iter, guess1, guess2, error_next);
        
        % Validate if the absolute residual error satisfies our accuracy threshold
        if abs(error_next) < tolerance
            fprintf('Convergence verified in %d iterations!\n', iter);
            final_solution = sol_next; % Lock in the valid structural state vectors
            converged = true;          % Raise success flag
            break;                     % Terminate search loop
        end
        
        % SHIFT HISTORICAL REGISTERS: Move the new values back to update the secant baseline
        guess1 = guess2;     error1 = error2;
        guess2 = next_guess; error2 = error_next;
    end
    
    % Hard stop check in case the method bounces out of control or diverges
    if ~converged, error('The secant system failed to resolve a root within iteration parameters.'); end
    
    % Execute the calculation output reporting and graphing tool functions
    print_and_plot_results(s_mesh, final_solution, L, 'Secant Method (Non-Linear Large Deflection)');
end

% =========================================================================
% HELPER UTILITY A: RUNGE-KUTTA 4TH ORDER STEP INTEGRATOR
% =========================================================================
function y = RK4(f, x, y0)
    y = zeros(length(y0), length(x)); % Pre-allocate state matrix size dimensions
    y(:, 1) = y0;                     % Establish initial boundary edge parameters at the wall (s=0)
    h = x(2) - x(1);                  % Spatial grid width resolution interval step size
    
    % Loop through the spatial grid to update vectors incrementally
    for i = 1:length(x)-1
        k1 = f(x(i), y(:, i));
        k2 = f(x(i) + h/2, y(:, i) + h*k1/2);
        k3 = f(x(i) + h/2, y(:, i) + h*k2/2);
        k4 = f(x(i) + h, y(:, i) + h*k3);
        
        % Compute a weighted average of the four slope estimates to update the next node step
        y(:, i+1) = y(:, i) + h*(k1 + 2*k2 + 2*k3 + k4)/6;
    end
end

% =========================================================================
% HELPER UTILITY B: DATA CONVERSION, REPORT PRINTING, AND GRAPH GENERATION
% =========================================================================
function print_and_plot_results(s_mesh, final_solution, L, method_title)
    vertical_y  = final_solution(1, :);   % Extract true vertical displacement profile (meters)
    slope_angle = final_solution(2, :);   % Extract structural rotation slope trajectory (radians)
    
    % Spatial reconstruction conversion: Compute actual horizontal coordinate layout
    % x = integral [ cos(theta) ds ]
    deformed_x = cumtrapz(s_mesh, cos(slope_angle)); 
    
    % Structural Shortening Profile mapping: Calculate inward tip movement length
    horiz_deflection_x = L - deformed_x(end);
    
    % Format print measurements to the terminal display window
    fprintf('\n================ CONVERGED PHYSICAL GEOMETRY METRICS ================\n');
    fprintf('Final Deflection Slope Angle:  %.4f degrees\n', rad2deg(slope_angle(end)));
    fprintf('Max Vertical Drop (y axis):    %.4f mm\n', vertical_y(end) * 1000);
    fprintf('Horizontal Shortening (x axis): %.4f mm\n', horiz_deflection_x * 1000);
    fprintf('=====================================================================\n');
    
    % Generate engineering plot visual displays
    figure('Color', [1 1 1]);
    plot(deformed_x, vertical_y, 'b-', 'LineWidth', 2.5); hold on;
    grid on; axis equal;
    title(method_title, 'FontSize', 11);
    xlabel('Deformed Horizontal Profile Position x (m)');
    ylabel('Downward Deflection Coordinate y (m)');
    set(gca, 'YDir', 'reverse'); % Reverses axis to point physical deflection downward
end