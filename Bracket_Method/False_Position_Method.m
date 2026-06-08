function False_Position_Method()
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
    % 4. THE SHOOTING METHOD MECHANICS (FALSE-POSITION BRACKETING)
    % =====================================================================
    lower_bound = -15.0;  % Under-guessed initial wall moment condition
    upper_bound = 5.0;    % Over-guessed initial wall moment condition
    
    % Run numerical simulations to gauge residual errors at both outer limits
    sol_low = RK4(f_system, s_mesh, [0; 0; lower_bound]); error_low = sol_low(3, end);
    sol_upp = RK4(f_system, s_mesh, [0; 0; upper_bound]); error_upp = sol_upp(3, end);
    
    if error_low * error_upp > 0
        error('Invalid initial domain brackets. Zero error root is not trapped.'); 
    end
    
    tolerance = 1e-9; max_iter = 50; converged = false;
    fprintf('\n--- FALSE-POSITION METHOD ITERATION LOG (NON-LINEAR) ---\n');
    fprintf('%-10s %-15s %-15s %-15s\n', 'Iter', 'Lower Bound', 'Upper Bound', 'Residual Error');
    
    for iter = 1:max_iter
        % REGULA FALSI LAW: Calculate zero-intercept crossing of the bracket line
        next_guess = upper_bound - (error_upp * (lower_bound - upper_bound)) / (error_low - error_upp);
        
        sol_next = RK4(f_system, s_mesh, [0; 0; next_guess]); error_next = sol_next(3, end);
        
        fprintf('%-10d %-15.4f %-15.4f %-15.4e\n', iter, lower_bound, upper_bound, error_next);
        
        if abs(error_next) < tolerance
            fprintf('Convergence verified in %d iterations!\n', iter);
            final_solution = sol_next; converged = true; break;
        end
        
        if (error_low * error_next) < 0
            upper_bound = next_guess; error_upp = error_next;
        else
            lower_bound = next_guess; error_low = error_next;
        end
    end
    if ~converged, error('The bracket system failed to resolve within iteration counts.'); end
    
    print_and_plot_results(s_mesh, final_solution, L, 'False-Position Method (Non-Linear)');
    
end  % <--- CRITICAL: Closes the main False_Position_Method function block

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