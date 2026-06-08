function Modified_Secant_Method()
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
    
    q = w_area * b;    % Linear load distribution (N/m)
    EI = E * I;        % Flexural Rigidity constant of the structure
    
    % =====================================================================
    % 2. NUMERICAL GRID MESH DISCRETIZATION
    % =====================================================================
    num_steps = 150;                  % Total spatial nodes across the beam
    s_mesh = linspace(0, L, num_steps); % Coordinate vectors tracking along arc length (s)
    
    % =====================================================================
    % 3. GEOMETRICALLY NON-LINEAR SYSTEM OF ODEs
    % =====================================================================
    f_system = @(s, Y) [Y(2); ...
                        Y(3); ...
                        -(q/EI)*(L - s)*cos(Y(2)) - (P/EI)*cos(Y(2))];
    
    % =====================================================================
    % 4. THE SHOOTING METHOD MECHANICS (MODIFIED SECANT ALGORITHM)
    % =====================================================================
    current_guess = -0.01; 
    delta = 1e-5; % Fractional modification step parameter
    
    tolerance = 1e-9; max_iter = 50; converged = false;
    
    fprintf('\n--- MODIFIED SECANT METHOD ITERATION LOG (NON-LINEAR) ---\n');
    fprintf('%-10s %-20s %-20s\n', 'Iter', 'Current Guess', 'Residual Error');
    
    for iter = 1:max_iter
        % Sample the baseline physical error behavior map response profile
        sol_current = RK4(f_system, s_mesh, [0; 0; current_guess]); 
        error_current = sol_current(3, end);
        
        fprintf('%-10d %-20.6f %-20.6e\n', iter, current_guess, error_current);
        
        if abs(error_current) < tolerance
            fprintf('Convergence verified in %d iterations!\n', iter);
            final_solution = sol_current; converged = true; break;
        end
        
        % FRACTIONAL MODIFICATION LAW: Scale step width change by current guess size
        if current_guess == 0
            dx = delta; 
        else
            dx = delta * current_guess; 
        end
        
        % Run a forward evaluation check on the modified step size path location
        sol_perturbed = RK4(f_system, s_mesh, [0; 0; current_guess + dx]); 
        error_perturbed = sol_perturbed(3, end);
        
        % Compute denominator tracking difference matrix profile
        denom = error_perturbed - error_current;
        if abs(denom) < 1e-15, break; end 
        
        % MODIFIED SECANT UPDATE STEP: Apply the fractional derivative update
        current_guess = current_guess - (error_current * dx) / denom;
    end
    if ~converged, error('The fractional modification secant loop failed to reach convergence.'); end
    
    print_and_plot_results(s_mesh, final_solution, L, 'Modified Secant Method (Non-Linear)');
    
end  % Closes the main Modified_Secant_Method function block

% =========================================================================
% HELPER UTILITY A: RUNGE-KUTTA 4TH ORDER STEP INTEGRATOR
% =========================================================================
function y = RK4(f, x, y0)
    y = zeros(length(y0), length(x)); 
    y(:, 1) = y0;                     
    h = x(2) - x(1);                  
    for i = 1:length(x)-1
        k1 = f(x(i), y(:, i));
        k2 = f(x(i) + h/2, y(:, i) + h*k1/2);
        k3 = f(x(i) + h/2, y(:, i) + h*k2/2);
        k4 = f(x(i) + h, y(:, i) + h*k3);
        y(:, i+1) = y(:, i) + h*(k1 + 2*k2 + 2*k3 + k4)/6;
    end
end % Closes RK4

% =========================================================================
% HELPER UTILITY B: DATA CONVERSION, REPORT PRINTING, AND GRAPH GENERATION
% =========================================================================
function print_and_plot_results(s_mesh, final_solution, L, method_title)
    vertical_y  = final_solution(1, :);   
    slope_angle = final_solution(2, :);   
    deformed_x = cumtrapz(s_mesh, cos(slope_angle)); 
    horiz_deflection_x = L - deformed_x(end);
    
    fprintf('\n================ CONVERGED PHYSICAL GEOMETRY METRICS ================\n');
    fprintf('Final Deflection Slope Angle:  %.4f degrees\n', rad2deg(slope_angle(end)));
    fprintf('Max Vertical Drop (y axis):    %.4f mm\n', vertical_y(end) * 1000);
    fprintf('Horizontal Shortening (x axis): %.4f mm\n', horiz_deflection_x * 1000);
    fprintf('=====================================================================\n');
    
    figure('Color', [1 1 1]);
    plot(deformed_x, vertical_y, 'b-', 'LineWidth', 2.5); hold on;
    grid on; axis equal;
    title(method_title, 'FontSize', 11);
    xlabel('Deformed Horizontal Profile Position x (m)');
    ylabel('Downward Deflection Coordinate y (m)');
    set(gca, 'YDir', 'reverse'); 
end % Closes print_and_plot_results