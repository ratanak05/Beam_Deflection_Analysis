function main_beam_deflection_secant()
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
    % 4. THE SHOOTING METHOD (Using SECANT Iteration)
    % =====================================================================
    guess1 = 0.0;
    guess2 = -0.01;
    
    sol1 = RK4(f_system, s_mesh, [0; 0; guess1]);
    error1 = sol1(3, end); 
    
    sol2 = RK4(f_system, s_mesh, [0; 0; guess2]);
    error2 = sol2(3, end); 
    
    tolerance = 1e-9;
    max_iter = 50;
    converged = false;
    
    fprintf('Starting Secant Root Search...\n');
    fprintf('%-10s %-15s %-15s %-15s\n', 'Iter', 'Old Guess (x1)', 'New Guess (x2)', 'Tip Moment Error');
    
    for iter = 1:max_iter
        if abs(error2 - error1) < 1e-15
            break; 
        end
        
        next_guess = guess2 - error2 * (guess2 - guess1) / (error2 - error1);
        
        sol_next = RK4(f_system, s_mesh, [0; 0; next_guess]);
        error_next = sol_next(3, end); 
        
        fprintf('%-10d %-15.4f %-15.4f %-15.4e\n', iter, guess1, guess2, error_next);
        
        if abs(error_next) < tolerance
            fprintf('Convergence achieved in %d iterations!\n', iter);
            final_solution = sol_next;
            converged = true;
            break;
        end
        
        guess1 = guess2;     error1 = error2;       
        guess2 = next_guess; error2 = error_next;   
    end
    
    if ~converged
        error('The Secant shooting method failed to converge.');
    end
    
    % =====================================================================
    % 5. EXTRACT ANGLE & CALCULATE X-DIRECTION DEFLECTION
    % =====================================================================
    vertical_y  = final_solution(1, :);   
    slope_angle = final_solution(2, :);   
    
    deformed_x = cumtrapz(s_mesh, cos(slope_angle));
    horiz_deflection_x = s_mesh - deformed_x;
    
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
    title('Deformed Cantilever Profile (Secant Solver)');
    xlabel('Horizontal Position x (m)');
    ylabel('Vertical Deflection y (m)');
    set(gca, 'YDir', 'reverse'); 
    
end  % <--- CRITICAL: THIS WAS MISSING. This closes your main function.

% =========================================================================
% 7. RUNGE-KUTTA 4 SYSTEM SOLVER
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
end % This explicitly closes your helper function.