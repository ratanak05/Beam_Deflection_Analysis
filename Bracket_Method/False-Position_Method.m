function main_beam_deflection_false_position()
    % =====================================================================
    % 1. PHYSICAL PARAMETERS (Exactly the same)
    % =====================================================================
    L  = 0.5;          % Length of the beam (meters)
    b  = 0.03;         % Width of cross-section (meters)
    t  = 0.01;         % Thickness of cross-section (meters)
    P  = 1000.0;       % Point load at free tip (N)
    w_area = 38500;    % Distributed load per area (N/m^2)
    E  = 200e9;        % Young's Modulus: 200 GPa
    I  = 2.50e-09;     % Moment of Inertia (m^4)
    
    q = w_area * b;    % Convert area load to linear load
    EI = E * I;        % Flexural rigidity
    
    % =====================================================================
    % 2. MESH DISCRETIZATION
    % =====================================================================
    num_steps = 150; 
    s_mesh = linspace(0, L, num_steps);
    
    % =====================================================================
    % 3. SYSTEM OF FIRST-ORDER ODEs (Anonymous Function)
    % =====================================================================
    f_system = @(s, Y) [Y(2); ...
                        Y(3); ...
                        -(q/EI)*(L - s) - (P/EI)];
                    
    % =====================================================================
    % 4. THE SHOOTING METHOD (Using FALSE-POSITION Iteration)
    % =====================================================================
    % False-Position requires initial bounds that trap the true root.
    % We guess initial values for the unknown wall constraint Y(3,1)
    lower_bound = -15.0;  
    upper_bound = 5.0;
    
    % Step 4.1: Compute the initial boundary errors using RK4
    sol_low = RK4(f_system, s_mesh, [0; 0; lower_bound]);
    error_low = sol_low(3, end); % Tip moment error for lower guess
    
    sol_upp = RK4(f_system, s_mesh, [0; 0; upper_bound]);
    error_upp = sol_upp(3, end); % Tip moment error for upper guess
    
    % Safety Check: Ensure the bounds actually bracket the root (opposite signs)
    if error_low * error_upp > 0
        error('The root is not bracketed by initial limits. Adjust lower_bound or upper_bound.');
    end
    
    tolerance = 1e-9;
    max_iter = 50;
    converged = false;
    
    fprintf('Starting False-Position Root Search...\n');
    fprintf('%-10s %-15s %-15s %-15s\n', 'Iter', 'Lower Bound', 'Upper Bound', 'Tip Moment Error');
    
    for iter = 1:max_iter
        % Step 4.2: FALSE-POSITION INTERPOLATION STEP
        % Finds the zero-intercept of the line linking the two boundary errors
        next_guess = upper_bound - (error_upp * (lower_bound - upper_bound)) / (error_low - error_upp);
        
        % Step 4.3: Integrate with the new guess
        sol_next = RK4(f_system, s_mesh, [0; 0; next_guess]);
        error_next = sol_next(3, end); % Target boundary is tip moment = 0
        
        fprintf('%-10d %-15.4f %-15.4f %-15.4e\n', iter, lower_bound, upper_bound, error_next);
        
        % Step 4.4: Check for convergence
        if abs(error_next) < tolerance
            fprintf('Convergence achieved in %d iterations!\n', iter);
            final_solution = sol_next;
            converged = true;
            break;
        end
        
        % Step 4.5: Update the active bracket boundaries (Sign Matching Strategy)
        % We evaluate the lower bound error against our guess error to collapse the interval
        if (error_low * error_next) < 0
            upper_bound = next_guess;
            error_upp   = error_next;
        else
            lower_bound = next_guess;
            error_low   = error_next;
        end
    end
    
    if ~converged
        error('The False-Position shooting method failed to converge.');
    end
    
    % =====================================================================
    % 5. EXTRACT ANGLE & CALCULATE X-DIRECTION DEFLECTION
    % =====================================================================
    vertical_y  = final_solution(1, :);   
    slope_angle = final_solution(2, :);   
    
    % True deformed horizontal coordinates
    deformed_x = cumtrapz(s_mesh, cos(slope_angle));
    
    % Horizontal displacement profile (u_x = s - x)
    horiz_deflection_x = s_mesh - deformed_x;
    
    % Print out results
    fprintf('\n--- RESULTS AT THE FREE TIP (s = L) ---\n');
    fprintf('Deflection Angle:          %.4f degrees\n', rad2deg(slope_angle(end)));
    fprintf('Vertical Deflection (y):    %.4f mm\n', vertical_y(end) * 1000);
    fprintf('Horizontal Deflection (x):  %.4f mm\n', horiz_deflection_x(end) * 1000);
    
    % =====================================================================
    % 6. PLOT THE TRUE 2D DEFORMED SHAPE
    % =====================================================================
    figure;
    plot(deformed_x, vertical_y, 'b-', 'LineWidth', 2.5);
    grid on;
    title('Deformed Cantilever Profile (False-Position Solver)');
    xlabel('Horizontal Position x (m)');
    ylabel('Vertical Deflection y (m)');
    set(gca, 'YDir', 'reverse'); 
end

% =========================================================================
% 7. RUNGE-KUTTA 4 SYSTEM SOLVER (Exactly your original implementation)
% =========================================================================
function y = RK4(f, x, y0)
    y = zeros(length(y0), length(x)); 
    y(:, 1) = y0; 
    h = x(2) - x(1); 
    n = length(x);
    for i = 1:n-1
        k1 = f(x(i), y(:, i));
        k2 = f(x(i) + h/2, y(:, i) + h*k1/2);
        k3 = f(x(i) + h/2, y(:, i) + h*k2/2);
        k4 = f(x(i) + h, y(:, i) + h*k3);
        y(:, i+1) = y(:, i) + h*(k1 + 2*k2 + 2*k3 + k4)/6;
    end
end